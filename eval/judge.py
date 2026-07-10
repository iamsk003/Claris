"""Held-out evaluation judge — the measurement instrument, never a selector.

This judge is deliberately separate from gate_2's critic:
  * its own prompt and rubric (it grades only two axes, ACCURACY and TONE, not the
    critic's four internal dimensions),
  * its own model id (a DIFFERENT Fireworks model from the one gate_2 runs),
  * it sees the ledger, the style contract, and ONE caption at a time, blind to whether
    the gates were on.

It is used only by ``eval/harness.py`` to score both ablation arms. Nothing in the shipped
pipeline imports it; it never influences selection or output. A gate that only improves the
score under the critic that drives it is not a gate — this judge is how that is verified.
"""

from __future__ import annotations

import json
import re
from typing import Optional

from pydantic import BaseModel, ValidationError, field_validator

from claris.core.generation.contracts import StyleContract
from claris.core.observability import EventSink, NullSink, log_llm_call
from claris.core.providers.base import ChatProvider
from claris.core.schema import EvidenceLedger, StyleName

# A different Fireworks model from gate_2's critic (VerificationConfig.critic_model,
# a Gemma). Independence is the point; eval/ is not part of the shipped agent.
DEFAULT_EVAL_JUDGE_MODEL = "accounts/fireworks/models/llama-v3p1-70b-instruct"

_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)

_SYSTEM = (
    "You are an impartial judge of video-captioning quality. You grade ONE caption "
    "on two axes:\n"
    "  ACCURACY (1-5): is every claim supported by the video evidence provided? Penalize "
    "any statement the evidence does not support.\n"
    "  TONE (1-5): does the caption convincingly read as the intended style, per its "
    "contract?\n"
    "You do not know how the caption was produced and must not reward or penalize length, "
    "citations, or format. Output STRICT JSON only: "
    '{"accuracy": n, "tone_fidelity": n, "accuracy_reason": "...", "tone_reason": "..."}.'
)


class JudgeScore(BaseModel):
    accuracy: float = 1.0
    tone_fidelity: float = 1.0
    accuracy_reason: str = ""
    tone_reason: str = ""
    judge_model: str = "held-out-judge"

    @field_validator("accuracy", "tone_fidelity")
    @classmethod
    def _bound(cls, v: float) -> float:
        return max(1.0, min(5.0, float(v)))

    @property
    def overall(self) -> float:
        """Accuracy and tone are graded; the held-out overall is their mean."""
        return (self.accuracy + self.tone_fidelity) / 2.0


def build_prompt(caption: str, style: StyleName, contract: StyleContract, ledger: EvidenceLedger) -> str:
    forbidden = "; ".join(contract.forbidden_moves) or "(none)"
    return (
        f"{ledger.to_prompt_block()}\n\n"
        f"STYLE: {style.value} — {contract.intent.strip()}\n"
        f"FORBIDDEN MOVES: {forbidden}\n\n"
        f'CAPTION TO GRADE:\n"{caption}"\n\n'
        "Return the JSON object only."
    )


def _parse(text: str) -> Optional[JudgeScore]:
    stripped = text.strip()
    cands = [stripped]
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
                return JudgeScore.model_validate(data)
            except ValidationError:
                continue
    return None


async def score_caption(
    caption: str,
    style: StyleName,
    contract: StyleContract,
    ledger: EvidenceLedger,
    provider: ChatProvider,
    *,
    model: Optional[str] = None,
    seed: int = 7,
    timeout_s: float = 60.0,
    sink: Optional[EventSink] = None,
    run_id: str = "eval",
) -> JudgeScore:
    """Grade one caption on accuracy and tone. Fails low (never crashes, never rewards)."""
    sink = sink or NullSink()
    try:
        result = await provider.complete(
            system=_SYSTEM,
            prompt=build_prompt(caption, style, contract, ledger),
            temperature=0.0, seed=seed, model=model, timeout_s=timeout_s, json_mode=True,
        )
        log_llm_call(sink, run_id, "held_out_judge", result,
                     event_id=f"{run_id}:{style.value}:judge_call")
        parsed = _parse(result.text)
        judge_model = model or getattr(provider, "model", "held-out-judge")
    except Exception:  # noqa: BLE001 — a judge failure must not crash the eval
        parsed, judge_model = None, model or getattr(provider, "model", "held-out-judge")
    if parsed is None:
        return JudgeScore(accuracy=1.0, tone_fidelity=1.0,
                          accuracy_reason="judge_unparseable", judge_model=judge_model)
    return parsed.model_copy(update={"judge_model": judge_model})


__all__ = ["JudgeScore", "score_caption", "build_prompt", "DEFAULT_EVAL_JUDGE_MODEL"]
