"""gate_1 (grounding/entailment) tests — corrected semantics.

Survival is threshold-based (support >= min_support). Contradiction is an immediate
fault regardless of score. Hedging (unused citations) is a scoring penalty, not an
elimination. Empty claim set / empty citation list -> 0.0 support (never 1.0, never a
ZeroDivisionError). The judge is mocked; the gate's decision logic and its fail-closed
and regenerate/degrade paths are what is under test.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from claris.core.verification.config import VerificationConfig
from claris.core.verification.hallucination import (
    ClaimAssessment,
    ClaimVerdict,
    evaluate_grounding,
    gate_1,
    ground_style,
    negative_constraint,
    support_score,
)
from claris.core.schema import (
    CaptionCandidate,
    EvidenceItem,
    EvidenceLedger,
    StyleName,
    VideoMeta,
)
from tests.fakes import FakeProvider

CFG = VerificationConfig()


def _ledger() -> EvidenceLedger:
    meta = VideoMeta(video_sha256="c" * 64, duration_s=30.0, has_audio=True)
    items = (
        EvidenceItem(id="E001", kind="visual", t_start=0.0, t_end=5.0,
                     content="A chef sears a steak in a cast-iron skillet.",
                     confidence=0.9, source_model="gemma-3-vlm"),
        EvidenceItem(id="E002", kind="speech", t_start=6.0, t_end=9.0,
                     content="Let it rest for about five minutes.",
                     confidence=0.88, source_model="faster-whisper"),
        EvidenceItem(id="E003", kind="audio_event", t_start=0.0, t_end=10.0,
                     content="Loud sizzling sound.", confidence=0.7,
                     source_model="librosa-heuristic"),
    )
    return EvidenceLedger(ledger_id="led", task_id="t", video_sha256="c" * 64,
                          video_meta=meta, items=items)


def _cand(text: str, cited: tuple[str, ...], cid: str = "cand_x") -> CaptionCandidate:
    return CaptionCandidate(candidate_id=cid, style=StyleName.FORMAL, text=text,
                            evidence_ids=cited, temperature=0.3, seed=1, model="fake")


def _assess(claim, verdict, ids=()):
    return ClaimAssessment(claim=claim, verdict=ClaimVerdict(verdict), supporting_ids=ids)


def _judge(assessments: list[dict]) -> FakeProvider:
    return FakeProvider(lambda *_: json.dumps(assessments))


def _run_gate(cand, judge):
    survivors, reports = asyncio.run(gate_1([cand], _ledger(), judge, CFG))
    return survivors, reports[0]


# --------------------------------------------------------------- support score (0.0 rule)


def test_support_score_empty_is_zero_not_one():
    assert support_score([]) == 0.0


def test_support_score_fraction():
    a = [_assess("x", "entailed", ("E001",)), _assess("y", "not_supported")]
    assert support_score(a) == 0.5


def test_evaluate_grounding_empty_citations_scores_zero():
    report = evaluate_grounding(_cand("A chef cooks.", ()), [], CFG)
    assert report.support_score == 0.0 and report.grounded is False


def test_all_entailed_precise_is_grounded():
    cand = _cand("A chef sears a steak.", ("E001",))
    report = evaluate_grounding(cand, [_assess("sears a steak", "entailed", ("E001",))], CFG)
    assert report.grounded is True and report.support_score == 1.0


# --------------------------------------------------------------- threshold + contradiction


def test_case1_contradiction_eliminated_regardless_of_score():
    cand = _cand("The chef poaches a fish in broth.", ("E001",))
    _, report = _run_gate(cand, _judge([{"claim": "poaches a fish", "verdict": "contradicted",
                                          "supporting_ids": ["E001"]}]))
    assert report.grounded is False and "contradicted_claims" in report.reasons


def test_contradiction_dominates_even_at_high_support():
    # 4 entailed + 1 contradicted = 0.8 support (at threshold), but contradiction eliminates.
    cand = _cand("...", ("E001",))
    assessments = [_assess(f"c{i}", "entailed", ("E001",)) for i in range(4)]
    assessments.append(_assess("bad", "contradicted", ("E001",)))
    report = evaluate_grounding(cand, assessments, CFG)
    assert report.grounded is False and "contradicted_claims" in report.reasons


def test_case4_unsupported_below_threshold_eliminated():
    cand = _cand("A chef sears a steak while a golden retriever watches.", ("E001",))
    _, report = _run_gate(cand, _judge([
        {"claim": "sears a steak", "verdict": "entailed", "supporting_ids": ["E001"]},
        {"claim": "a golden retriever watches", "verdict": "not_supported", "supporting_ids": []},
    ]))
    assert report.grounded is False and report.support_score == 0.5
    assert "unsupported_claims" in report.reasons and "below_min_support" in report.reasons


def test_partial_unsupported_above_threshold_survives():
    # 4 of 5 entailed = 0.8 >= min_support -> survives despite one unsupported claim.
    cand = _cand("...", ("E001",))
    assessments = [_assess(f"c{i}", "entailed", ("E001",)) for i in range(4)]
    assessments.append(_assess("weak", "not_supported"))
    report = evaluate_grounding(cand, assessments, CFG)
    assert report.grounded is True and report.support_score == 0.8


# --------------------------------------------------------------- hedging = penalty


def test_case6_hedging_penalized_not_eliminated():
    # One entailed claim supported by E001; E002/E003 cited but unused.
    # 1.0 - 0.05*2 = 0.90 >= 0.8 -> survives, but flagged and penalized.
    cand = _cand("Some things occur.", ("E001", "E002", "E003"))
    _, report = _run_gate(cand, _judge([
        {"claim": "some things occur", "verdict": "entailed", "supporting_ids": ["E001"]},
    ]))
    assert report.grounded is True
    assert report.raw_support == 1.0 and report.support_score == 0.9
    assert "hedging_unused_citations" in report.reasons
    assert set(report.unused_citations) == {"E002", "E003"}


def test_excessive_hedging_falls_below_threshold():
    # One entailed claim but five unused citations: 1.0 - 0.05*5 = 0.75 < 0.8 -> eliminated.
    cand = _cand("...", tuple(f"E{i:03d}" for i in range(1, 7)))
    report = evaluate_grounding(cand, [_assess("x", "entailed", ("E001",))], CFG)
    assert report.grounded is False and report.support_score == 0.75


# --------------------------------------------------------------- fail closed


def test_fail_closed_on_judge_exception():
    def boom(*_):
        raise RuntimeError("judge down")

    _, report = _run_gate(_cand("A chef sears a steak.", ("E001",)), FakeProvider(boom))
    assert report.grounded is False and report.failed_closed is True


def test_fail_closed_on_unparseable_judge_output():
    _, report = _run_gate(_cand("A chef sears a steak.", ("E001",)), FakeProvider(lambda *_: "no"))
    assert report.grounded is False and report.failed_closed is True


# --------------------------------------------------------------- regenerate / degrade path


def _keyword_judge() -> FakeProvider:
    # Contradicts any caption mentioning 'poach'; entails otherwise.
    def resp(system, prompt, t, s):
        if "poach" in prompt.lower():
            return json.dumps([{"claim": "poaches", "verdict": "contradicted", "supporting_ids": ["E001"]}])
        return json.dumps([{"claim": "sears", "verdict": "entailed", "supporting_ids": ["E001"]}])
    return FakeProvider(resp)


def test_ground_style_regenerates_and_recovers():
    original = [_cand("The chef poaches a fish.", ("E001",), "orig")]
    grounded_replacement = [_cand("A chef sears a steak.", ("E001",), "regen")]

    async def regen(constraint):
        assert "poach" in constraint.lower()  # negative constraint quotes the bad claim
        return grounded_replacement

    outcome = asyncio.run(
        ground_style(StyleName.FORMAL, original, _ledger(), _keyword_judge(), CFG, regenerate_fn=regen)
    )
    assert outcome.regenerated is True and outcome.degraded_ungrounded is False
    assert [c.candidate_id for c in outcome.survivors] == ["regen"]


def test_ground_style_degrades_to_best_original_when_regen_also_fails():
    # Two bad originals with different support; both regenerations also bad -> emit best.
    originals = [_cand("The chef poaches a fish.", ("E001",), "orig_low")]

    async def regen(constraint):
        return [_cand("The chef poaches again.", ("E001",), "regen_bad")]

    outcome = asyncio.run(
        ground_style(StyleName.FORMAL, originals, _ledger(), _keyword_judge(), CFG, regenerate_fn=regen)
    )
    assert outcome.degraded_ungrounded is True
    assert len(outcome.survivors) == 1  # never empty: a flawed caption beats an absent one


def test_negative_constraint_quotes_claims():
    from claris.core.verification.hallucination import GroundingReport

    r = GroundingReport(candidate_id="x", support_score=0.0,
                        contradicted_claims=("a dog appears",),
                        unsupported_claims=("it is raining",))
    text = negative_constraint([r])
    assert "a dog appears" in text and "it is raining" in text


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
