"""Runtime-only perception output for CLARIS v2.

The frozen ``EvidenceLedger`` stays textual evidence only. Keyframe images travel
separately, alongside the ledger, in this dataclass — never inside the ledger — so the
single downstream reasoning call can attach the frames without the schema ever carrying
image bytes.
"""

from __future__ import annotations

from dataclasses import dataclass

from claris.core.perception.shots import Keyframe
from claris.core.schema import EvidenceLedger


@dataclass(frozen=True)
class PerceptionBundle:
    """The textual ledger plus the ordered keyframes for one clip."""

    ledger: EvidenceLedger
    keyframes: tuple[Keyframe, ...] = ()
