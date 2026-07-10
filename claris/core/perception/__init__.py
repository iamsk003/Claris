"""Perception — turns a video file into an immutable EvidenceLedger.

Content-aware keyframe sampling (not uniform), plus four more signal channels most
competitors ignore: speech (faster-whisper), on-screen text (OCR), non-speech audio
events, and Gemma 3 VLM keyframe understanding grounded on the OCR.

Public entry point:
    build_ledger(task, cfg, vision_provider=...) -> EvidenceLedger

Heavy library imports (OpenCV, scenedetect, PaddleOCR, librosa, faster-whisper) are
lazy and every heavy stage is injectable, so the assembly logic runs and is tested
offline with zero network calls.
"""

from claris.core.perception.config import PerceptionConfig
from claris.core.perception.ledger import (
    assemble_ledger,
    assign_ids,
    build_ledger,
    compute_modality_flags,
    interval_union_coverage,
)

__all__ = [
    "PerceptionConfig",
    "build_ledger",
    "assemble_ledger",
    "assign_ids",
    "compute_modality_flags",
    "interval_union_coverage",
]
