"""gate_2 — the Gemma critic that scores and selects.

Each surviving candidate is scored 1-5 on four dimensions — accuracy, tone_fidelity,
style_distinctness, naturalness — with the weights coming from VerificationConfig, not
constants. The style contract is passed into the prompt so the critic grades tone against
the contract, not against its own notion of the style.

Bias controls: candidates are scored in a randomized (seeded) order, and each is scored
twice with different seeds and averaged. Per-dimension reasoning strings are preserved for
the frontend Judge View. Selection is argmax on the config-weighted overall, with a
deterministic tie-break so the winner never depends on presentation order.
"""

from __future__ import annotations

import json
import random
import re
from typing import Optional

from pydantic import BaseModel, ValidationError, field_validator

from claris.core.generation.contracts import StyleContract
from claris.core.observability import EventSink, NullSink, log_llm_call
from claris.core.providers.base import ChatProvider
from claris.core.schema import CaptionCandidate, CritiqueScore, EvidenceLedger, RunEvent
from claris.core.verification.config import CriticWeights, VerificationConfig

_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def _clamp15(v: float) -> float:
    return max(1.0, min(5.0, float(v)))


class _CriticRaw(BaseModel):
    """One critic pass, before averaging. Scores clamped to [1, 5]."""

    accuracy: float = 1.0
    tone_fidelity: float = 1.0
    style_distinctness: float = 1.0
    naturalness: float = 1.0
    accuracy_reason: str = ""
    tone_reason: str = ""
    distinctness_reason: str = ""
    naturalness_reason: str = ""

    @field_validator("accuracy", "tone_fidelity", "style_distinctness", "naturalness")
    @classmethod
    def _bound(cls, v: float) -> float:
        return _clamp15(v)


def weighted_overall(
    accuracy: float, tone_fidelity: float, style_distinctness: float, naturalness: float,
    weights: CriticWeights,
) -> float:
    """Config-weighted aggregate on the 1-5 scale. Pure."""
    return (
        weights.accuracy * accuracy
        + weights.tone_fidelity * tone_fidelity
        + weights.style_distinctness * style_distinctness
        + weights.naturalness * naturalness
    )


def _system_prompt(contract: StyleContract) -> str:
    forbidden = "; ".join(contract.forbidden_moves) or "(none)"
    return (
        "You are a strict caption critic. Score ONE caption from 1 to 5 on four dimensions:\n"
        "- accuracy: is every statement supported by the video evidence provided?\n"
        f"- tone_fidelity: does it match THIS style contract? Grade against the contract, "
        f"not your own idea of the style.\n"
        f"    style: {contract.name.value}\n"
        f"    intent: {contract.intent.strip()}\n"
        f"    voice: {contract.voice.strip()}\n"
        f"    forbidden moves: {forbidden}\n"
        "- style_distinctness: how clearly does it read as this style and not a neighbour?\n"
        "- naturalness: does it read as fluent, non-templated human writing?\n"
        'Respond with STRICT JSON: {"accuracy": n, "tone_fidelity": n, '
        '"style_distinctness": n, "naturalness": n, "accuracy_reason": "...", '
        '"tone_reason": "...", "distinctness_reason": "...", "naturalness_reason": "..."}. '
        "Scores are numbers 1-5. Output the JSON object only."
    )


def _user_prompt(candidate: CaptionCandidate, ledger: EvidenceLedger) -> str:
    return (
        f"{ledger.to_prompt_block()}\n\n"
        f'Caption ({candidate.style.value}): "{candidate.text}"\n\n'
        "Score this caption. Respond with the JSON object only."
    )


def _parse_raw(text: str) -> Optional[_CriticRaw]:
    stripped = text.strip()
    cands = [stripped]
    if stripped.startswith("```"):
        cands.append(re.sub(r"^json\s*", "", stripped.strip("`"), flags=re.IGNORECASE).strip())
    m = _JSON_OBJ_RE.search(stripped)
    if m:
        cands.append(m.group(0))
    for c in cands:
        try:
            data = json.loads(c)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(data, dict):
            try:
                return _CriticRaw.model_validate(data)
            except ValidationError:
                continue
    return None


def _average(a: _CriticRaw, b: _CriticRaw) -> _CriticRaw:
    return _CriticRaw(
        accuracy=(a.accuracy + b.accuracy) / 2,
        tone_fidelity=(a.tone_fidelity + b.tone_fidelity) / 2,
        style_distinctness=(a.style_distinctness + b.style_distinctness) / 2,
        naturalness=(a.naturalness + b.naturalness) / 2,
        # Reasons are qualitative; keep the first pass's, which is deterministic.
        accuracy_reason=a.accuracy_reason,
        tone_reason=a.tone_reason,
        distinctness_reason=a.distinctness_reason,
        naturalness_reason=a.naturalness_reason,
    )


async def critique_candidate(
    candidate: CaptionCandidate,
    ledger: EvidenceLedger,
    contract: StyleContract,
    critic: ChatProvider,
    cfg: Optional[VerificationConfig] = None,
    *,
    sink: Optional[EventSink] = None,
    run_id: str = "verification",
) -> CritiqueScore:
    """Score one candidate: two seeds averaged. A broken critic scores low, never crashes."""
    cfg = cfg or VerificationConfig()
    system = _system_prompt(contract)
    user = _user_prompt(candidate, ledger)

    passes: list[_CriticRaw] = []
    critic_model = getattr(critic, "model", "gemma-critic")
    sink = sink or NullSink()
    for i, seed in enumerate(cfg.critic_seeds):
        try:
            result = await critic.complete(
                system=system, prompt=user, temperature=cfg.critic_temperature,
                seed=seed, timeout_s=cfg.critic_timeout_s, json_mode=True,
            )
            critic_model = result.model
            log_llm_call(sink, run_id, "gate_2", result,
                         event_id=f"{run_id}:{candidate.candidate_id}:gate2_call_{i}")
            raw = _parse_raw(result.text)
        except Exception:  # noqa: BLE001 — a bad critic call must not sink the batch
            raw = None
        passes.append(raw or _CriticRaw(accuracy_reason="critic_unavailable"))

    avg = _average(passes[0], passes[1]) if len(passes) >= 2 else passes[0]
    overall = weighted_overall(
        avg.accuracy, avg.tone_fidelity, avg.style_distinctness, avg.naturalness, cfg.weights
    )
    score = CritiqueScore(
        accuracy=avg.accuracy, tone_fidelity=avg.tone_fidelity,
        style_distinctness=avg.style_distinctness, naturalness=avg.naturalness,
        overall=round(overall, 4),
        accuracy_reason=avg.accuracy_reason, tone_reason=avg.tone_reason,
        distinctness_reason=avg.distinctness_reason, naturalness_reason=avg.naturalness_reason,
        critic_model=critic_model,
    )
    (sink or NullSink()).emit(
        RunEvent(
            run_id=run_id, event_id=f"{run_id}:{candidate.candidate_id}:critique",
            stage="verification", event_type="candidate_scored",
            model=critic_model,
            payload={"candidate_id": candidate.candidate_id, "style": candidate.style.value,
                     "overall": score.overall},
        )
    )
    return score


async def score_and_select(
    candidates: list[CaptionCandidate],
    ledger: EvidenceLedger,
    contract: StyleContract,
    critic: ChatProvider,
    cfg: Optional[VerificationConfig] = None,
    *,
    order_seed: int = 0,
    sink: Optional[EventSink] = None,
    run_id: str = "verification",
) -> tuple[Optional[CaptionCandidate], Optional[CritiqueScore], list[tuple[CaptionCandidate, CritiqueScore]]]:
    """Score all candidates (in randomized order) and return the argmax winner.

    Returns (winner, winner_score, all_scored). Order randomization is a bias control;
    because scoring is per-candidate independent, the winner is invariant to it.
    """
    cfg = cfg or VerificationConfig()
    if not candidates:
        return None, None, []

    order = list(range(len(candidates)))
    random.Random(order_seed).shuffle(order)

    scored_by_index: dict[int, CritiqueScore] = {}
    for i in order:
        scored_by_index[i] = await critique_candidate(
            candidates[i], ledger, contract, critic, cfg, sink=sink, run_id=run_id
        )

    scored = [(candidates[i], scored_by_index[i]) for i in range(len(candidates))]
    # argmax on overall; deterministic tie-break by candidate_id.
    winner, winner_score = max(scored, key=lambda cs: (cs[1].overall, cs[0].candidate_id))
    return winner, winner_score, scored


__all__ = ["weighted_overall", "critique_candidate", "score_and_select"]
