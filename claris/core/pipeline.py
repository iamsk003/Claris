"""The core pipeline. One engine, one event stream.

    probe -> shots -> (speech || ocr || audio_events || vision) -> ledger
          -> generate(4 styles x N) -> gate_1 -> gate_2 -> select -> gate_3 -> TaskResult

Perception is delegated to ``claris.core.perception.build_ledger`` (which already runs the
concurrent detectors). This module owns generation + the three verification gates and the
assembly of the final ``TaskResult``.

Every stage transition emits a ``RunEvent`` on the injected sink. That single JSONL stream
is what the agent persists and what the API will later publish over WebSocket — there is no
second progress mechanism. Providers are injected, so the whole pipeline runs offline in
tests with fakes and makes zero network calls.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from claris.core.generation import GenerationConfig, StyleContractRegistry, generate_all
from claris.core.observability import EventSink, NullSink
from claris.core.perception import PerceptionConfig, build_ledger
from claris.core.providers.base import ChatProvider, VisionProvider
from claris.core.schema import (
    CaptionCandidate,
    CritiqueScore,
    EvidenceLedger,
    ProviderTier,
    RunEvent,
    StyledCaption,
    StyleName,
    Task,
    TaskResult,
    utcnow,
)
from claris.core.verification import (
    VerificationConfig,
    enforce_separation,
    ground_style,
    score_and_select,
)
from claris.core.verification.separation import EmbedFn


@dataclass
class Providers:
    """The inference backends the pipeline needs. Perception's vision is optional."""

    gen_provider: ChatProvider
    judge: ChatProvider
    critic: ChatProvider
    vision_provider: Optional[VisionProvider] = None
    embed_fn: Optional[EmbedFn] = None


def _emit(sink: EventSink, run_id: str, task_id: str, event_type: str, **payload) -> None:
    sink.emit(
        RunEvent(
            run_id=run_id,
            event_id=f"{run_id}:{event_type}",
            stage="pipeline",
            event_type=event_type,
            task_id=task_id,
            payload=payload,
        )
    )


def _prior_block(prior: dict[StyleName, str]) -> str:
    """The 'already written, do not overlap' block for sequential generation."""
    if not prior:
        return ""
    lines = [
        "ALREADY WRITTEN for other styles of THIS SAME video. Do NOT overlap with any of "
        "them in phrasing, observation, or joke structure — choose a different detail from "
        "the evidence and a different angle:"
    ]
    for style, text in prior.items():
        lines.append(f"[{style.value}] {text}")
    return "\n".join(lines)


def _template_caption(style: StyleName, ledger: EvidenceLedger) -> StyledCaption:
    """Deterministic last resort: never emit an empty caption for a style."""
    items = sorted(ledger.items, key=lambda it: (it.t_start, it.id))
    lead = next((it for it in items if it.kind.value == "visual"), items[0] if items else None)
    if lead is not None:
        text = "This clip shows " + (lead.content[0].lower() + lead.content[1:])
        evidence = (lead.id,)
    else:
        text = "This clip could not be captioned from the available evidence."
        evidence = ()
    return StyledCaption(
        style=style, text=text, evidence_ids=evidence,
        provider_tier=ProviderTier.TEMPLATE,
        degraded=True, degradation_reason="template_fallback",
    )


def _styled_from(
    candidate: CaptionCandidate, score: Optional[CritiqueScore], degraded_ungrounded: bool
) -> StyledCaption:
    return StyledCaption(
        style=candidate.style,
        text=candidate.text,
        candidate_id=candidate.candidate_id,
        evidence_ids=candidate.evidence_ids,
        score=score,
        provider_tier=candidate.provider_tier,
        degraded=degraded_ungrounded,
        degradation_reason="ungrounded_best_of_batch" if degraded_ungrounded else None,
        degraded_ungrounded=degraded_ungrounded,
    )


async def _generate_for_style(
    style: StyleName, note: str, ledger: EvidenceLedger, task: Task, providers: Providers,
    registry: StyleContractRegistry, gen_cfg: GenerationConfig, sink: EventSink, run_id: str,
) -> list[CaptionCandidate]:
    """Regenerate one style's candidates with an extra prompt note (constraint/contrast)."""
    t2 = task.model_copy(update={"notes": ((task.notes or "") + "\n" + note).strip()})
    batch = await generate_all(
        ledger, t2, providers.gen_provider, registry=registry, config=gen_cfg,
        sink=sink, run_id=run_id, styles=(style,),
    )
    return batch[style]


async def run_from_ledger(
    ledger: EvidenceLedger,
    task: Task,
    providers: Providers,
    *,
    gen_config: Optional[GenerationConfig] = None,
    ver_config: Optional[VerificationConfig] = None,
    registry: Optional[StyleContractRegistry] = None,
    sink: Optional[EventSink] = None,
    use_gates: bool = True,
    run_id: Optional[str] = None,
) -> TaskResult:
    """Run generation + verification over an existing ledger. The harness entry point."""
    gen_config = gen_config or GenerationConfig()
    ver_config = ver_config or VerificationConfig()
    registry = registry or StyleContractRegistry()
    sink = sink or NullSink()
    run_id = run_id or f"pipe_{task.task_id}_{utcnow().strftime('%Y%m%dT%H%M%S')}"
    styles = task.styles

    _emit(sink, run_id, task.task_id, "generation_start", n=gen_config.n,
          use_gates=use_gates, sequential=gen_config.sequential)

    captions: dict[StyleName, StyledCaption] = {}

    def _regen_for(style: StyleName):
        async def regen(constraint: str) -> list[CaptionCandidate]:
            return await _generate_for_style(
                style, constraint, ledger, task, providers, registry, gen_config, sink, run_id
            )
        return regen

    if not use_gates:
        # Ablation baseline: take the first candidate per style, no verification at all.
        candidates = await generate_all(
            ledger, task, providers.gen_provider, registry=registry, config=gen_config,
            sink=sink, run_id=run_id,
        )
        _emit(sink, run_id, task.task_id, "generation_done",
              counts={s.value: len(candidates[s]) for s in styles})
        for style in styles:
            cands = candidates[style]
            captions[style] = (
                _styled_from(cands[0], None, False) if cands else _template_caption(style, ledger)
            )
        _emit(sink, run_id, task.task_id, "pipeline_done", gated=False)
        return TaskResult(
            task_id=task.task_id, run_id=run_id, video_sha256=ledger.video_sha256,
            ledger_id=ledger.ledger_id, captions=captions,
            degraded=any(c.degraded for c in captions.values()),
        )

    selected: dict[StyleName, tuple[CaptionCandidate, CritiqueScore, bool]] = {}

    if gen_config.sequential:
        # PREVENTION: generate styles in order, each prompt carrying the already-chosen
        # captions as a "do not overlap" block. Ground + select each before the next.
        prior_texts: dict[StyleName, str] = {}
        order = [s for s in gen_config.sequential_order if s in styles]
        for style in order:
            cands = await _generate_for_style(
                style, _prior_block(prior_texts), ledger, task, providers, registry,
                gen_config, sink, run_id,
            )
            outcome = await ground_style(style, cands, ledger, providers.judge, ver_config,
                                         regenerate_fn=_regen_for(style), sink=sink, run_id=run_id)
            winner, score, _all = await score_and_select(
                list(outcome.survivors), ledger, registry.get(style), providers.critic,
                ver_config, sink=sink, run_id=run_id,
            )
            if winner is not None and score is not None:
                selected[style] = (winner, score, outcome.degraded_ungrounded)
                prior_texts[style] = winner.text
        _emit(sink, run_id, task.task_id, "generation_done", sequential=True,
              order=[s.value for s in order])
        _emit(sink, run_id, task.task_id, "gate_2_done",
              selected={s.value: round(selected[s][1].overall, 3) for s in selected})
    else:
        candidates = await generate_all(
            ledger, task, providers.gen_provider, registry=registry, config=gen_config,
            sink=sink, run_id=run_id,
        )
        _emit(sink, run_id, task.task_id, "generation_done",
              counts={s.value: len(candidates[s]) for s in styles})

        # gate_1 — grounding, with per-style regenerate/degrade.
        outcomes = {}
        for style in styles:
            outcomes[style] = await ground_style(
                style, candidates[style], ledger, providers.judge, ver_config,
                regenerate_fn=_regen_for(style), sink=sink, run_id=run_id,
            )
        _emit(sink, run_id, task.task_id, "gate_1_done",
              survivors={s.value: len(outcomes[s].survivors) for s in styles},
              degraded=[s.value for s in styles if outcomes[s].degraded_ungrounded])

        # gate_2 — critic scores and selects the argmax per style.
        for style in styles:
            outcome = outcomes[style]
            winner, score, _all = await score_and_select(
                list(outcome.survivors), ledger, registry.get(style), providers.critic,
                ver_config, sink=sink, run_id=run_id,
            )
            if winner is not None and score is not None:
                selected[style] = (winner, score, outcome.degraded_ungrounded)
        _emit(sink, run_id, task.task_id, "gate_2_done",
              selected={s.value: round(selected[s][1].overall, 3) for s in selected})

    # gate_3 — tone separation over the selected winners.
    sep_input = {s: (c, sc) for s, (c, sc, _d) in selected.items()}

    async def replace_fn(style: StyleName, contrast: str):
        cands = await _generate_for_style(
            style, contrast, ledger, task, providers, registry, gen_config, sink, run_id
        )
        outcome2 = await ground_style(style, cands, ledger, providers.judge, ver_config,
                                      sink=sink, run_id=run_id)
        if not outcome2.survivors:
            return None
        w2, s2, _ = await score_and_select(
            list(outcome2.survivors), ledger, registry.get(style), providers.critic,
            ver_config, sink=sink, run_id=run_id,
        )
        return (w2, s2) if w2 is not None else None

    separated = await enforce_separation(
        sep_input, ver_config, replace_fn=replace_fn,
        embed_fn=providers.embed_fn, sink=sink, run_id=run_id,
    )
    _emit(sink, run_id, task.task_id, "gate_3_done")

    for style in styles:
        if style in separated:
            winner, score = separated[style]
            captions[style] = _styled_from(winner, score, selected[style][2])
        else:
            captions[style] = _template_caption(style, ledger)

    _emit(sink, run_id, task.task_id, "pipeline_done", gated=True)
    return TaskResult(
        task_id=task.task_id, run_id=run_id, video_sha256=ledger.video_sha256,
        ledger_id=ledger.ledger_id, captions=captions,
        degraded=any(c.degraded for c in captions.values()),
    )


async def run_pipeline(
    task: Task,
    providers: Providers,
    *,
    perception_config: Optional[PerceptionConfig] = None,
    gen_config: Optional[GenerationConfig] = None,
    ver_config: Optional[VerificationConfig] = None,
    registry: Optional[StyleContractRegistry] = None,
    sink: Optional[EventSink] = None,
    use_gates: bool = True,
    run_id: Optional[str] = None,
    **perception_kwargs,
) -> TaskResult:
    """Full pipeline from a video file: perception then generation + verification."""
    sink = sink or NullSink()
    run_id = run_id or f"pipe_{task.task_id}_{utcnow().strftime('%Y%m%dT%H%M%S')}"

    # vision_provider may be None: perception then omits the VLM visual layer and builds a
    # ledger from speech + OCR + audio + motion. A degraded ledger beats an absent caption.
    _emit(sink, run_id, task.task_id, "perception_start", vision=providers.vision_provider is not None)
    ledger = await build_ledger(
        task, perception_config, vision_provider=providers.vision_provider,
        sink=sink, **perception_kwargs,
    )
    _emit(sink, run_id, task.task_id, "perception_done",
          items=len(ledger.items), coverage=ledger.coverage)

    return await run_from_ledger(
        ledger, task, providers, gen_config=gen_config, ver_config=ver_config,
        registry=registry, sink=sink, use_gates=use_gates, run_id=run_id,
    )


__all__ = ["Providers", "run_from_ledger", "run_pipeline"]
