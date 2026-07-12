"""CLARIS v2 reasoning: one multimodal call turns a clip into four captions.

The model receives the ordered keyframes plus a STRUCTURED-JSON evidence object
(transcript / ocr / audio_events / motion, each timestamped) and returns ONE JSON object
holding a summary, a temporal event list, the four styled captions, and factual metadata
(``grounded`` + which modalities it used). No rejection sampling, no critic, no separation
pass — grounding is a property of this single grounded prompt plus a zero-cost id check.

The provider is a generic multimodal ``reasoning_provider`` (Kimi / Qwen / GLM / MiniMax /
Gemma …); nothing here is model-specific.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable, Optional

from claris.core.observability import EventSink, NullSink
from claris.core.perception.bundle import PerceptionBundle
from claris.core.perception.config import PerceptionConfig
from claris.core.providers.base import VisionProvider
from claris.core.schema import (
    EvidenceKind,
    EvidenceLedger,
    ProviderTier,
    RunEvent,
    StyledCaption,
    StyleName,
    Task,
    TaskResult,
    utcnow,
)

# JSON caption key <-> canonical StyleName.
_STYLE_KEYS: dict[StyleName, str] = {
    StyleName.FORMAL: "formal",
    StyleName.SARCASTIC: "sarcastic",
    StyleName.HUMOROUS_TECH: "tech_humor",
    StyleName.HUMOROUS_NON_TECH: "everyday_humor",
}

_SYSTEM = (
    "You are a precise video analyst. You are given a handful of keyframes sampled IN ORDER "
    "from ONE short video, plus a JSON object of timestamped evidence extracted from it "
    "(transcript, on-screen text, audio events, motion). Reason about the video AS A WHOLE — "
    "what happens, who/what is involved, and how it unfolds over time. Do NOT describe the "
    "frames as separate images. Ground every statement in the frames and the given evidence; "
    "never invent facts, names, brands, or numbers that are not supported. Reply with ONE "
    "JSON object and nothing else."
)

# Concise per-style voice guides (distilled from generation/styles/*.yaml). Kept inline so
# the single prompt is self-contained and cheap to build.
_STYLE_GUIDE = (
    "captions must all describe the SAME real content but sound distinct:\n"
    "- formal: neutral third-person record of what occurs; no opinion, no contractions, "
    "no exclamation.\n"
    "- sarcastic: dry, wry understatement or mock-praise; still factually correct.\n"
    "- tech_humor: playful joke using software/engineering metaphors.\n"
    "- everyday_humor: light relatable everyday joke, no tech jargon.\n"
    "Each caption is one or two sentences, natural and concise."
)


def _kind_entries(ledger: EvidenceLedger, kind: EvidenceKind, key: str) -> list[dict]:
    return [
        {"time": round(it.t_start, 2), key: it.content}
        for it in ledger.by_kind(kind)
    ]


def build_evidence_json(ledger: EvidenceLedger) -> dict:
    """The structured evidence object handed to the reasoning model. Pure."""
    return {
        "transcript": _kind_entries(ledger, EvidenceKind.SPEECH, "text"),
        "ocr": _kind_entries(ledger, EvidenceKind.OCR, "text"),
        "audio_events": _kind_entries(ledger, EvidenceKind.AUDIO_EVENT, "event"),
        "motion": _kind_entries(ledger, EvidenceKind.MOTION, "event"),
    }


def build_prompt(evidence: dict, n_frames: int) -> str:
    """The user-turn prompt: evidence JSON + the required output shape. Pure."""
    schema = (
        '{"summary": str, "events": [{"time": number, "description": str}], '
        '"captions": {"canonical": str, "formal": str, "sarcastic": str, '
        '"tech_humor": str, "everyday_humor": str}, "grounded": bool, '
        '"modalities": {"speech_used": bool, "ocr_used": bool, "audio_used": bool}}'
    )
    return (
        f"{n_frames} keyframes from the video are attached, in chronological order.\n\n"
        f"EVIDENCE (timestamps in seconds):\n{json.dumps(evidence, ensure_ascii=False)}\n\n"
        f"First write a canonical caption: a faithful, concise account of the whole video. "
        f"Then rewrite it in each of the four styles. {_STYLE_GUIDE}\n\n"
        f"Set \"grounded\" false if the evidence and frames are too thin to caption "
        f"reliably. Set each \"modalities\" flag by whether you actually used that channel.\n\n"
        f"Return ONLY this JSON object:\n{schema}"
    )


def _json_candidates(text: str) -> list[str]:
    """Raw object, a ```json-fenced object, or the first balanced {...} span."""
    out = [text]
    stripped = text.strip()
    if stripped.startswith("```"):
        out.append(re.sub(r"^json\s*", "", stripped.strip("`"), flags=re.IGNORECASE))
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        out.append(text[start : end + 1])
    return out


def parse_reasoning(text: str) -> Optional[dict]:
    """Best-effort parse of the single reasoning JSON. Pure; None if unparseable."""
    for candidate in _json_candidates(text):
        try:
            data = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(data, dict) and isinstance(data.get("captions"), dict):
            return data
    return None


def reasoning_to_result(
    task: Task,
    ledger: EvidenceLedger,
    data: Optional[dict],
    *,
    tier: ProviderTier,
    model: str,
    run_id: str,
) -> TaskResult:
    """Map the reasoning JSON onto a TaskResult of four StyledCaptions. Pure.

    Grounding without a verification pass: every caption is supported by the full ledger's
    evidence ids; ``grounded=false`` (or a missing/failed parse) flags the affected styles
    degraded rather than dropping them.
    """
    evidence_ids = tuple(it.id for it in ledger.items)
    captions_in = (data or {}).get("captions", {}) if isinstance(data, dict) else {}
    canonical = str(captions_in.get("canonical", "")).strip()
    grounded = bool(data.get("grounded", True)) if isinstance(data, dict) else False

    captions: dict[StyleName, StyledCaption] = {}
    for style in task.styles:
        text = str(captions_in.get(_STYLE_KEYS[style], "")).strip() or canonical
        if not text:
            captions[style] = _degraded_caption(style, ledger, "reasoning_unparsed")
            continue
        degraded = not grounded
        captions[style] = StyledCaption(
            style=style,
            text=text,
            evidence_ids=evidence_ids,
            provider_tier=tier,
            degraded=degraded,
            degradation_reason="model_reported_ungrounded" if degraded else None,
            degraded_ungrounded=degraded,
        )

    return TaskResult(
        task_id=task.task_id,
        run_id=run_id,
        video_sha256=ledger.video_sha256,
        ledger_id=ledger.ledger_id,
        captions=captions,
        degraded=any(c.degraded for c in captions.values()),
    )


def _degraded_caption(style: StyleName, ledger: EvidenceLedger, reason: str) -> StyledCaption:
    """Last resort: a deterministic caption so a style is never absent."""
    lead = next((it for it in ledger.items), None)
    if lead is not None:
        text = "This clip shows " + (lead.content[0].lower() + lead.content[1:])
        evidence = (lead.id,)
    else:
        text = "This clip could not be captioned from the available evidence."
        evidence = ()
    return StyledCaption(
        style=style, text=text, evidence_ids=evidence,
        provider_tier=ProviderTier.TEMPLATE, degraded=True, degradation_reason=reason,
    )


def _load_images(bundle: PerceptionBundle) -> list[bytes]:  # pragma: no cover - reads files
    return [Path(kf.image_path).read_bytes() for kf in bundle.keyframes]


def _emit(sink: EventSink, run_id: str, task_id: str, event_type: str, **payload) -> None:
    sink.emit(RunEvent(run_id=run_id, event_id=f"{run_id}:{event_type}", stage="reasoning",
                       event_type=event_type, task_id=task_id, payload=payload))


async def reason_over_clip(
    bundle: PerceptionBundle,
    task: Task,
    provider: VisionProvider,
    *,
    cfg: Optional[PerceptionConfig] = None,
    sink: Optional[EventSink] = None,
    run_id: Optional[str] = None,
    image_loader: Optional[Callable[[PerceptionBundle], list[bytes]]] = None,
) -> TaskResult:
    """The whole of CLARIS v2 generation: one multimodal call -> four captions."""
    cfg = cfg or PerceptionConfig()
    sink = sink or NullSink()
    run_id = run_id or f"reason_{task.task_id}"
    image_loader = image_loader or _load_images

    ledger = bundle.ledger
    evidence = build_evidence_json(ledger)
    images = image_loader(bundle)
    prompt = build_prompt(evidence, len(images))
    _emit(sink, run_id, task.task_id, "reasoning_start",
          frames=len(images), evidence_counts={k: len(v) for k, v in evidence.items()})

    try:
        result = await provider.complete(
            system=_SYSTEM, prompt=prompt, images=images,
            temperature=cfg.vision_temperature, seed=cfg.seed,
            max_tokens=cfg.vision_max_tokens, timeout_s=cfg.vision_timeout_s, json_mode=True,
        )
        data = parse_reasoning(result.text)
        tier, model = result.provider_tier, result.model
    except Exception as exc:  # noqa: BLE001 — one failed call degrades, never crashes the task
        # TEMP instrumentation: surface why the single call failed. Behavior unchanged.
        resp = getattr(exc, "response", None)
        body = getattr(resp, "text", None)
        _emit(
            sink, run_id, task.task_id, "reasoning_failed",
            exc_type=type(exc).__name__,
            exc_msg=str(exc)[:500],
            http_status=getattr(resp, "status_code", None),
            response_body=body[:1000] if isinstance(body, str) else None,
            error=repr(exc)[:200],
        )
        data, tier, model = None, ProviderTier.TEMPLATE, "unavailable"

    task_result = reasoning_to_result(task, ledger, data, tier=tier, model=model, run_id=run_id)
    _emit(sink, run_id, task.task_id, "reasoning_done",
          grounded=(data or {}).get("grounded") if isinstance(data, dict) else False,
          modalities=(data or {}).get("modalities") if isinstance(data, dict) else None,
          degraded=task_result.degraded, parsed=data is not None)
    return task_result


__all__ = [
    "build_evidence_json",
    "build_prompt",
    "parse_reasoning",
    "reasoning_to_result",
    "reason_over_clip",
]
