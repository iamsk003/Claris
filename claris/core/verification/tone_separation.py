"""Deprecated shim. Tone separation (gate_3) lives in ``separation.py``."""

from __future__ import annotations

from claris.core.verification.separation import enforce_separation, pairwise_similarities

__all__ = ["enforce_separation", "pairwise_similarities"]
