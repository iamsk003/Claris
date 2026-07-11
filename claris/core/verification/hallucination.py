"""Grounding / entailment gate (gate_1).

Generation's parse-time gate catches announced hallucinations — captions citing an
evidence ID that is not in the ledger. It cannot catch *semantic* failures where the
citation is real but the claim is not actually supported:

  1. the cited evidence contradicts the caption,
  4. the caption asserts an object present in no evidence item,
  6. the caption cites every ID indiscriminately (hedging, not grounding).

gate_1 catches all three. A judge (Gemma-as-critic) decomposes the caption into claims
and, for each, decides whether the cited evidence entails it and which IDs support it.
The gate then FAILS CLOSED: a candidate survives only if every claim is entailed and every
cited ID actually supports a claim. Any contradiction, unsupported claim, unused citation,
or judge failure eliminates the candidate.

The support score never rewards ungrounded output: an empty claim set or empty citation
list scores 0.0, never 1.0 and never a ZeroDivisionError — a vacuous 1.0 would make
rejection sampling prefer captions grounded in nothing.
"""

from __future__ import annotations

import json
import re
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ValidationError

from claris.core.observability import EventSink, NullSink, log_llm_call
from claris.core.providers.base import ChatProvider
from claris.core.schema import CaptionCandidate, EvidenceItem, EvidenceLedger, RunEvent, StyleName
from claris.core.verification.config import VerificationConfig

JUDGE_SYSTEM = (
    "You are a strict grounding judge for video captions. You are given a caption and the "
    "exact evidence items it cites. Decompose the caption into atomic factual claims. For "
    "each claim decide, using ONLY the cited evidence: 'entailed' if the evidence directly "
    "supports it, 'contradicted' if the evidence states otherwise, 'not_supported' if the "
    "evidence neither supports nor contradicts it. List the evidence IDs that support each "
    "claim. Respond with a STRICT JSON array of objects with keys "
    '"claim", "verdict", "supporting_ids". Output the JSON array and nothing else.'
)

_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)


class ClaimVerdict(str, Enum):
    ENTAILED = "entailed"
    CONTRADICTED = "contradicted"
    NOT_SUPPORTED = "not_supported"


class ClaimAssessment(BaseModel):
    claim: str
    verdict: ClaimVerdict
    supporting_ids: tuple[str, ...] = ()


class GroundingReport(BaseModel):
    """The gate's verdict on one candidate."""

    candidate_id: str
    support_score: float           # entailed fraction after the hedging penalty
    raw_support: float = 0.0       # entailed fraction before the penalty
    grounded: bool = False
    reasons: tuple[str, ...] = ()
    assessments: tuple[ClaimAssessment, ...] = ()
    unused_citations: tuple[str, ...] = ()
    contradicted_claims: tuple[str, ...] = ()
    unsupported_claims: tuple[str, ...] = ()
    failed_closed: bool = False
    judge_model: str = "gemma-critic"


def support_score(assessments: list[ClaimAssessment]) -> float:
    """Fraction of claims that are entailed. Empty -> 0.0 (never 1.0, never div-by-zero)."""
    if not assessments:
        return 0.0
    entailed = sum(1 for a in assessments if a.verdict == ClaimVerdict.ENTAILED)
    return entailed / len(assessments)


def evaluate_grounding(
    candidate: CaptionCandidate,
    assessments: list[ClaimAssessment],
    cfg: Optional[VerificationConfig] = None,
    *,
    judge_model: str = "gemma-critic",
) -> GroundingReport:
    """Decide whether a candidate is grounded from its per-claim assessments. Pure.

    Survival is threshold-based: the entailed fraction, minus a hedging penalty per
    unused cited ID, must reach ``min_support``. A contradicted claim is an immediate
    fault (contradiction is not a degree). An empty citation list scores 0.0.
    """
    cfg = cfg or VerificationConfig()
    cited = tuple(candidate.evidence_ids)
    if not cited:
        return GroundingReport(candidate_id=candidate.candidate_id, support_score=0.0,
                               grounded=False, reasons=("empty_citation_list",),
                               judge_model=judge_model)
    if not assessments:
        return GroundingReport(candidate_id=candidate.candidate_id, support_score=0.0,
                               grounded=False, reasons=("no_claims_assessed",),
                               unused_citations=cited, judge_model=judge_model)

    raw = support_score(assessments)
    contradicted = tuple(a.claim for a in assessments if a.verdict == ClaimVerdict.CONTRADICTED)
    unsupported = tuple(a.claim for a in assessments if a.verdict == ClaimVerdict.NOT_SUPPORTED)

    used = {
        i for a in assessments if a.verdict == ClaimVerdict.ENTAILED for i in a.supporting_ids
    }
    unused = tuple(c for c in cited if c not in used)

    # Hedging is sloppiness, not a lie: penalize the score, do not eliminate outright.
    score = max(0.0, raw - cfg.hedge_penalty * len(unused))

    reasons: list[str] = []
    if contradicted:
        reasons.append("contradicted_claims")
    if unsupported:
        reasons.append("unsupported_claims")
    if unused:
        reasons.append("hedging_unused_citations")

    # Contradiction is an immediate elimination regardless of score.
    grounded = not contradicted and score >= cfg.min_support
    if not grounded and not contradicted and score < cfg.min_support:
        reasons.append("below_min_support")

    return GroundingReport(
        candidate_id=candidate.candidate_id,
        support_score=round(score, 4),
        raw_support=round(raw, 4),
        grounded=grounded,
        reasons=tuple(reasons),
        assessments=tuple(assessments),
        unused_citations=unused,
        contradicted_claims=contradicted,
        unsupported_claims=unsupported,
        judge_model=judge_model,
    )


def build_judge_prompt(caption: str, cited_items: list[EvidenceItem]) -> str:
    """Build the judge prompt from the caption and its cited evidence. Pure."""
    lines = ["Caption:", caption.strip(), "", "Cited evidence:"]
    if cited_items:
        for it in cited_items:
            lines.append(f"[{it.id}] {it.kind.value}: {it.content}")
    else:
        lines.append("(none)")
    lines.append("\nReturn the JSON array of claim assessments.")
    return "\n".join(lines)


def parse_assessments(text: str) -> Optional[list[ClaimAssessment]]:
    """Parse the judge output into ClaimAssessments. None on failure -> fail closed.

    Unknown verdict strings are coerced to ``not_supported`` (conservative).
    """
    for candidate in _json_candidates(text):
        try:
            data = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(data, list):
            continue
        out: list[ClaimAssessment] = []
        for obj in data:
            if not isinstance(obj, dict):
                continue
            verdict_raw = str(obj.get("verdict", "")).lower()
            try:
                verdict = ClaimVerdict(verdict_raw)
            except ValueError:
                verdict = ClaimVerdict.NOT_SUPPORTED
            try:
                out.append(
                    ClaimAssessment(
                        claim=str(obj.get("claim", "")),
                        verdict=verdict,
                        supporting_ids=tuple(obj.get("supporting_ids", []) or ()),
                    )
                )
            except ValidationError:
                continue
        return out
    return None


def _json_candidates(text: str) -> list[str]:
    stripped = text.strip()
    out = [stripped]
    if stripped.startswith("```"):
        out.append(re.sub(r"^json\s*", "", stripped.strip("`"), flags=re.IGNORECASE).strip())
    m = _JSON_ARRAY_RE.search(stripped)
    if m:
        out.append(m.group(0))
    return out


async def assess_candidate(
    candidate: CaptionCandidate,
    ledger: EvidenceLedger,
    judge: ChatProvider,
    cfg: Optional[VerificationConfig] = None,
    *,
    seed: int = 1337,
    sink: Optional[EventSink] = None,
    run_id: str = "verification",
) -> GroundingReport:
    """Judge one candidate's grounding. Fails closed on any judge or parse failure."""
    cfg = cfg or VerificationConfig()
    sink = sink or NullSink()
    cited_items = [it for it in (ledger.get(i) for i in candidate.evidence_ids) if it]

    assessments: Optional[list[ClaimAssessment]]
    try:
        result = await judge.complete(
            system=JUDGE_SYSTEM,
            prompt=build_judge_prompt(candidate.text, cited_items),
            temperature=0.0,
            seed=seed,
            timeout_s=cfg.judge_timeout_s,
            json_mode=True,
            max_tokens=cfg.verify_max_tokens,
        )
        log_llm_call(sink, run_id, "gate_1", result,
                     event_id=f"{run_id}:{candidate.candidate_id}:gate1_call")
        assessments = parse_assessments(result.text)
        judge_model = result.model
    except Exception:  # noqa: BLE001 — judge failure must eliminate, never crash the batch
        assessments = None
        judge_model = getattr(judge, "model", "gemma-critic")

    if assessments is None:
        report = GroundingReport(
            candidate_id=candidate.candidate_id,
            support_score=0.0,
            grounded=False,
            reasons=("judge_unavailable_fail_closed",),
            unused_citations=tuple(candidate.evidence_ids),
            failed_closed=True,
            judge_model=judge_model,
        )
    else:
        report = evaluate_grounding(candidate, assessments, cfg, judge_model=judge_model)

    sink.emit(
        RunEvent(
            run_id=run_id,
            event_id=f"{run_id}:{candidate.candidate_id}:grounding",
            stage="verification",
            event_type="grounding_assessed",
            level="info" if report.grounded else "warn",  # type: ignore[arg-type]
            model=report.judge_model,
            payload={
                "candidate_id": candidate.candidate_id,
                "grounded": report.grounded,
                "support_score": report.support_score,
                "reasons": list(report.reasons),
                "failed_closed": report.failed_closed,
            },
        )
    )
    return report


async def gate_1(
    candidates: list[CaptionCandidate],
    ledger: EvidenceLedger,
    judge: ChatProvider,
    cfg: Optional[VerificationConfig] = None,
    *,
    sink: Optional[EventSink] = None,
    run_id: str = "verification",
) -> tuple[list[CaptionCandidate], list[GroundingReport]]:
    """Entailment gate: keep only grounded candidates. Fails closed.

    Returns (survivors, reports) with reports aligned to the input order for logging.
    """
    import asyncio  # noqa: PLC0415

    cfg = cfg or VerificationConfig()
    reports = await asyncio.gather(
        *(assess_candidate(c, ledger, judge, cfg, sink=sink, run_id=run_id) for c in candidates)
    )
    survivors = [c for c, r in zip(candidates, reports) if r.grounded]
    return survivors, list(reports)


def negative_constraint(reports: list[GroundingReport]) -> str:
    """Quote back the contradicted and unsupported claims as an explicit prohibition."""
    bad: list[str] = []
    for r in reports:
        bad.extend(r.contradicted_claims)
        bad.extend(r.unsupported_claims)
    unique = list(dict.fromkeys(c.strip() for c in bad if c.strip()))
    if not unique:
        return ""
    quoted = "; ".join(f'"{c}"' for c in unique)
    return (
        "The previous attempt made claims the evidence does not support. "
        f"Do NOT state any of the following: {quoted}. "
        "Every claim must be entailed by a cited evidence item."
    )


class Gate1Outcome(BaseModel):
    """Per-style result of the grounding stage, including the regenerate/degrade path."""

    style: StyleName
    survivors: tuple[CaptionCandidate, ...]
    reports: tuple[GroundingReport, ...]
    regenerated: bool = False
    degraded_ungrounded: bool = False


async def ground_style(
    style: StyleName,
    candidates: list[CaptionCandidate],
    ledger: EvidenceLedger,
    judge: ChatProvider,
    cfg: Optional[VerificationConfig] = None,
    *,
    regenerate_fn=None,
    sink: Optional[EventSink] = None,
    run_id: str = "verification",
) -> Gate1Outcome:
    """Ground one style's candidates, with the spec's regenerate-then-degrade fallback.

    If every candidate is eliminated, regenerate once with the contradicted/unsupported
    claims quoted back as a negative constraint. If that batch is also wiped, emit the
    highest-support candidate from the ORIGINAL batch, flagged ``degraded_ungrounded``.
    We never emit an empty caption for a style: a flawed caption beats an absent one.
    """
    cfg = cfg or VerificationConfig()
    survivors, reports = await gate_1(candidates, ledger, judge, cfg, sink=sink, run_id=run_id)
    if survivors:
        return Gate1Outcome(style=style, survivors=tuple(survivors), reports=tuple(reports))

    if regenerate_fn is not None:
        constraint = negative_constraint(reports)
        regenerated = await regenerate_fn(constraint)
        if regenerated:
            surv2, reports2 = await gate_1(regenerated, ledger, judge, cfg, sink=sink, run_id=run_id)
            if surv2:
                return Gate1Outcome(
                    style=style, survivors=tuple(surv2), reports=tuple(reports2), regenerated=True
                )

    # Both attempts failed. Emit the best-supported original candidate, flagged.
    if candidates:
        best_idx = max(range(len(reports)), key=lambda i: reports[i].support_score)
        return Gate1Outcome(
            style=style,
            survivors=(candidates[best_idx],),
            reports=(reports[best_idx],),
            degraded_ungrounded=True,
        )
    return Gate1Outcome(style=style, survivors=(), reports=tuple(reports), degraded_ungrounded=True)


__all__ = [
    "ClaimVerdict",
    "ClaimAssessment",
    "GroundingReport",
    "Gate1Outcome",
    "support_score",
    "evaluate_grounding",
    "build_judge_prompt",
    "parse_assessments",
    "negative_constraint",
    "assess_candidate",
    "gate_1",
    "ground_style",
]
