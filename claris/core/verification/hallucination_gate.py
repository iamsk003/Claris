"""Deprecated shim. The grounding/entailment gate lives in ``hallucination.py``."""

from __future__ import annotations

from claris.core.verification.hallucination import assess_candidate, evaluate_grounding, gate_1

__all__ = ["gate_1", "assess_candidate", "evaluate_grounding"]
