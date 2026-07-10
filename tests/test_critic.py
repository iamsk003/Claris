"""gate_2 (critic) tests. Deterministic winner; order-invariant selection."""

from __future__ import annotations

import asyncio
import json

from claris.core.generation import StyleContractRegistry
from claris.core.verification.config import VerificationConfig
from claris.core.verification.critic import score_and_select, weighted_overall
from claris.core.schema import (
    CaptionCandidate,
    EvidenceItem,
    EvidenceLedger,
    StyleName,
    VideoMeta,
)
from tests.fakes import FakeProvider

CFG = VerificationConfig()
CONTRACT = StyleContractRegistry().get(StyleName.FORMAL)


def _ledger() -> EvidenceLedger:
    meta = VideoMeta(video_sha256="c" * 64, duration_s=30.0, has_audio=True)
    items = (
        EvidenceItem(id="E001", kind="visual", t_start=0.0, t_end=5.0,
                     content="A chef sears a steak.", confidence=0.9, source_model="vlm"),
    )
    return EvidenceLedger(ledger_id="l", task_id="t", video_sha256="c" * 64,
                          video_meta=meta, items=items)


def _cand(text: str, cid: str) -> CaptionCandidate:
    return CaptionCandidate(candidate_id=cid, style=StyleName.FORMAL, text=text,
                            evidence_ids=("E001",), temperature=0.3, seed=1, model="fake")


def _score(acc, tone, dist, nat) -> str:
    return json.dumps({
        "accuracy": acc, "tone_fidelity": tone, "style_distinctness": dist, "naturalness": nat,
        "accuracy_reason": "a", "tone_reason": "t", "distinctness_reason": "d",
        "naturalness_reason": "n",
    })


def _critic_by_text(mapping: dict[str, str]) -> FakeProvider:
    # Returns the scored JSON keyed on which caption text appears in the prompt.
    def resp(system, prompt, t, s):
        for text, out in mapping.items():
            if text in prompt:
                return out
        return _score(1, 1, 1, 1)
    return FakeProvider(resp)


def test_weighted_overall_uses_config_weights():
    ov = weighted_overall(5, 1, 1, 1, CFG.weights)
    assert ov == 5 * 0.40 + 1 * 0.35 + 1 * 0.15 + 1 * 0.10


def test_deterministic_winner():
    good = _cand("The strong caption.", "good")
    weak = _cand("The weak caption.", "weak")
    critic = _critic_by_text({
        "The strong caption.": _score(5, 5, 5, 5),
        "The weak caption.": _score(2, 2, 2, 2),
    })
    winner, score, scored = asyncio.run(
        score_and_select([weak, good], _ledger(), CONTRACT, critic, CFG)
    )
    assert winner.candidate_id == "good" and score.overall == 5.0
    assert len(scored) == 2


def test_reversing_presentation_order_does_not_change_winner():
    good = _cand("The strong caption.", "good")
    weak = _cand("The weak caption.", "weak")
    critic = _critic_by_text({
        "The strong caption.": _score(5, 5, 4, 5),
        "The weak caption.": _score(2, 3, 2, 2),
    })
    w1, _, _ = asyncio.run(score_and_select([good, weak], _ledger(), CONTRACT, critic, CFG,
                                            order_seed=1))
    w2, _, _ = asyncio.run(score_and_select([weak, good], _ledger(), CONTRACT, critic, CFG,
                                            order_seed=999))
    assert w1.candidate_id == w2.candidate_id == "good"


def test_two_seeds_are_averaged():
    # Critic returns the same score for both seeds (FakeProvider ignores seed), so the
    # average equals the single-pass value. Reasons are preserved for the Judge View.
    cand = _cand("A caption.", "c")
    critic = _critic_by_text({"A caption.": _score(4, 3, 5, 2)})
    _, score, _ = asyncio.run(score_and_select([cand], _ledger(), CONTRACT, critic, CFG))
    assert score.accuracy == 4 and score.tone_fidelity == 3
    assert score.accuracy_reason == "a" and score.tone_reason == "t"


def test_unparseable_critic_scores_low_not_crash():
    cand = _cand("A caption.", "c")
    _, score, _ = asyncio.run(
        score_and_select([cand], _ledger(), CONTRACT, FakeProvider(lambda *_: "broken"), CFG)
    )
    assert score.overall == 1.0  # fail low, never crash
