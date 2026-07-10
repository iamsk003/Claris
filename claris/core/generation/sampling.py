"""Rejection sampling over candidates (STUB — do not implement yet).

Samples N candidates per style at varied temperature, defers scoring to the critic
(claris.core.verification.critic), and returns the argmax on the critic objective.
"""

from __future__ import annotations

from claris.core.schema import CaptionCandidate, EvidenceLedger, StyleName


def sample_candidates(
    ledger: EvidenceLedger, style: StyleName, n: int
) -> list[CaptionCandidate]:  # pragma: no cover
    """Sample N candidates for a style. Not implemented."""
    raise NotImplementedError("generation.sampling.sample_candidates is a stub")
