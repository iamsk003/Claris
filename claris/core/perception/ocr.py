"""On-screen text extraction. The cheapest accuracy gain nobody else is taking.

OCR every keyframe, drop low-confidence and sub-pixel-noise boxes, and deduplicate
strings across frames. This recovers signage, slides, UI text, brand names,
lower-thirds and subtitles — the concrete referents that make humorous_tech land.

Filtering and dedup are pure and unit tested; the PaddleOCR engine is injected via
``ocr_fn`` so tests run without the model.
"""

from __future__ import annotations

import re
from typing import Callable, Optional

from pydantic import BaseModel

from claris.core.perception.config import PerceptionConfig
from claris.core.perception.shots import Keyframe
from claris.core.schema import EvidenceItem

SOURCE_MODEL = "paddleocr"


class OcrBox(BaseModel):
    """One detected text box on a single frame."""

    text: str
    confidence: float
    area_frac: float  # box area as a fraction of the frame area


def normalize_text(text: str) -> str:
    """Dedup key: collapse whitespace and lowercase. Pure."""
    return re.sub(r"\s+", " ", text).strip().lower()


def filter_boxes(boxes: list[OcrBox], cfg: PerceptionConfig) -> list[OcrBox]:
    """Drop boxes below the confidence floor or the minimum area fraction. Pure."""
    return [
        b
        for b in boxes
        if b.confidence >= cfg.ocr_min_confidence
        and b.area_frac >= cfg.ocr_min_area_frac
        and b.text.strip()
    ]


def dedup_ocr(
    frame_boxes: list[tuple[float, OcrBox]], source_model: str = SOURCE_MODEL
) -> list[EvidenceItem]:
    """Collapse repeated strings across frames into one item each. Pure.

    Keys on normalized text; keeps the earliest appearance for timing and the highest
    confidence seen. Input is (t, box) pairs from all keyframes.
    """
    best: dict[str, tuple[float, OcrBox]] = {}
    for t, box in frame_boxes:
        key = normalize_text(box.text)
        if not key:
            continue
        cur = best.get(key)
        if cur is None or box.confidence > cur[1].confidence:
            keep_t = min(t, cur[0]) if cur else t
            best[key] = (keep_t, box)
        else:
            best[key] = (min(t, cur[0]), cur[1])

    items: list[EvidenceItem] = []
    ordered = sorted(best.items(), key=lambda kv: kv[1][0])
    for i, (_key, (t, box)) in enumerate(ordered, start=1):
        items.append(
            EvidenceItem(
                id=f"ocr_{i:03d}",
                kind="ocr",
                t_start=t,
                t_end=t,
                content=box.text.strip(),
                confidence=round(box.confidence, 3),
                source_model=source_model,
            )
        )
    return items


def run_ocr(
    keyframes: list[Keyframe],
    cfg: Optional[PerceptionConfig] = None,
    *,
    ocr_fn: Optional[Callable[[str], list[OcrBox]]] = None,
) -> tuple[list[EvidenceItem], dict[float, list[str]]]:
    """OCR each keyframe, filter, dedup. Returns (items, per-frame strings for VLM).

    ``ocr_fn(image_path) -> list[OcrBox]`` is injected in tests. The second return
    value maps keyframe timestamp -> the OCR strings on it, so the VLM can be grounded
    on what is actually written on each frame.
    """
    cfg = cfg or PerceptionConfig()
    # PaddleOCR has been removed; the vision model reads on-screen text directly from frames.
    # This stays callable with an injected ``ocr_fn`` (used by tests); with none it contributes
    # nothing, so the ledger carries no OCR items and evidence["ocr"] is an empty list.
    if ocr_fn is None:
        return [], {}

    frame_boxes: list[tuple[float, OcrBox]] = []
    per_frame: dict[float, list[str]] = {}
    for kf in keyframes:
        boxes = filter_boxes(ocr_fn(kf.image_path), cfg)
        per_frame[kf.t_mid] = [b.text.strip() for b in boxes]
        frame_boxes.extend((kf.t_mid, b) for b in boxes)

    return dedup_ocr(frame_boxes), per_frame
