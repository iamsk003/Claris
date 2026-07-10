"""Gemma 3 VLM over the selected keyframes.

Keyframes are sent as a batch with their timestamps and the OCR strings already found
on each, so the VLM grounds its description rather than guessing. We ask for a strict
JSON array — one object per frame with fixed keys — parse it defensively, and retry
once with a repair prompt if the first parse fails.

Prompt building and parsing are pure; the VLM call goes through an injected
``VisionProvider`` so tests never touch the network.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ValidationError

from claris.core.perception.config import PerceptionConfig
from claris.core.perception.shots import Keyframe
from claris.core.providers.base import VisionProvider
from claris.core.schema import EvidenceItem, RunEvent
from claris.core.observability import EventSink, NullSink, log_llm_call

SOURCE_MODEL = "gemma-3-vlm"

_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)

_SYSTEM = (
    "You describe video keyframes for a captioning system. You are given frames with "
    "their timestamps and any text detected on them by OCR. Ground every statement in "
    "what is visible; do not invent detail. Respond with a STRICT JSON array, one object "
    "per frame in the given order, each with exactly these keys: "
    '"setting", "subjects", "actions", "objects", "mood", "notable_detail". '
    '"subjects", "actions" and "objects" are arrays of short strings; the rest are '
    "strings. Output the JSON array and nothing else."
)


class FrameDescription(BaseModel):
    setting: str = ""
    subjects: list[str] = []
    actions: list[str] = []
    objects: list[str] = []
    mood: str = ""
    notable_detail: str = ""


def build_prompt(keyframes: list[Keyframe], ocr_by_frame: dict[float, list[str]]) -> str:
    """Build the user prompt listing each frame's timestamp and OCR strings. Pure."""
    lines = ["Frames:"]
    for i, kf in enumerate(keyframes, start=1):
        ocr = ocr_by_frame.get(kf.t_mid, [])
        ocr_str = "; ".join(ocr) if ocr else "(none)"
        lines.append(f"Frame {i} at {kf.t_mid:.1f}s. OCR text on frame: {ocr_str}")
    lines.append(
        f"\nReturn a JSON array of exactly {len(keyframes)} objects, one per frame."
    )
    return "\n".join(lines)


def parse_descriptions(text: str, expected: int) -> Optional[list[FrameDescription]]:
    """Parse the VLM output into FrameDescriptions. Returns None on failure. Pure.

    Accepts the raw array, a ```json-fenced array, or the first ``[...]`` span, and
    pads/truncates to ``expected`` length only when at least one object parsed.
    """
    for candidate in _json_candidates(text):
        try:
            data = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(data, dict):
            data = [data]  # single-frame calls return a bare object, not an array
        if not isinstance(data, list):
            continue
        descs: list[FrameDescription] = []
        for obj in data:
            if not isinstance(obj, dict):
                continue
            try:
                descs.append(FrameDescription.model_validate(obj))
            except ValidationError:
                descs.append(FrameDescription())
        if descs:
            if len(descs) < expected:
                descs += [FrameDescription()] * (expected - len(descs))
            return descs[:expected]
    return None


def _json_candidates(text: str) -> list[str]:
    stripped = text.strip()
    out = [stripped]
    if stripped.startswith("```"):
        inner = re.sub(r"^json\s*", "", stripped.strip("`"), flags=re.IGNORECASE)
        out.append(inner.strip())
    m = _JSON_ARRAY_RE.search(stripped)
    if m:
        out.append(m.group(0))
    m2 = re.search(r"\{.*\}", stripped, re.DOTALL)  # single-object fallback
    if m2:
        out.append(m2.group(0))
    return out


def compose_content(desc: FrameDescription) -> str:
    """Flatten a frame description into one grounded sentence for the ledger. Pure."""
    parts: list[str] = []
    if desc.setting:
        parts.append(desc.setting.rstrip("."))
    if desc.subjects:
        parts.append("Subjects: " + ", ".join(desc.subjects))
    if desc.actions:
        parts.append("Actions: " + ", ".join(desc.actions))
    if desc.objects:
        parts.append("Objects: " + ", ".join(desc.objects))
    if desc.notable_detail:
        parts.append("Notable: " + desc.notable_detail.rstrip("."))
    if desc.mood:
        parts.append("Mood: " + desc.mood.rstrip("."))
    # Empty -> "" (not a "no detail" placeholder): an empty description should NOT become a
    # visual evidence item that pollutes the ledger and gets picked up as a caption.
    return ". ".join(p for p in parts if p) + "." if parts else ""


def descriptions_to_items(
    descs: list[FrameDescription],
    keyframes: list[Keyframe],
    cfg: PerceptionConfig,
    source_model: str = SOURCE_MODEL,
) -> list[EvidenceItem]:
    """One visual EvidenceItem per keyframe with a non-empty description. Pure."""
    items: list[EvidenceItem] = []
    idx = 0
    for desc, kf in zip(descs, keyframes):
        content = compose_content(desc)
        if not content:
            continue  # drop empty descriptions instead of emitting "no detail" noise
        idx += 1
        items.append(
            EvidenceItem(
                id=f"visual_{idx:03d}",
                kind="visual",
                t_start=kf.t_mid,
                t_end=kf.t_mid,
                content=content,
                confidence=cfg.vision_confidence,
                source_model=source_model,
            )
        )
    return items


def _load_images(keyframes: list[Keyframe]) -> list[bytes]:  # pragma: no cover - reads files
    return [Path(kf.image_path).read_bytes() for kf in keyframes]


async def describe_keyframes(
    keyframes: list[Keyframe],
    ocr_by_frame: dict[float, list[str]],
    provider: VisionProvider,
    cfg: Optional[PerceptionConfig] = None,
    *,
    sink: Optional[EventSink] = None,
    run_id: str = "perception",
    images: Optional[list[bytes]] = None,
) -> list[EvidenceItem]:
    """Describe each keyframe with the VLM, one image per call.

    Multi-image batches are unreliable — the VLM merges them or returns an empty object —
    so we describe frames individually. A frame that errors or fails to parse is skipped
    (no "no detail" placeholder); an all-failed clip yields an empty visual layer.
    """
    cfg = cfg or PerceptionConfig()
    sink = sink or NullSink()
    if not keyframes:
        return []
    imgs = images if images is not None else _load_images(keyframes)

    good_descs: list[FrameDescription] = []
    good_kfs: list[Keyframe] = []
    for i, (kf, img) in enumerate(zip(keyframes, imgs), start=1):
        try:
            result = await provider.complete(
                system=_SYSTEM, prompt=build_prompt([kf], ocr_by_frame), images=[img],
                temperature=cfg.vision_temperature, seed=cfg.seed, model=cfg.vision_model,
                max_tokens=cfg.vision_max_tokens, timeout_s=cfg.vision_timeout_s, json_mode=True,
            )
        except Exception as exc:  # noqa: BLE001 — one bad frame must not sink the stage
            sink.emit(RunEvent(run_id=run_id, event_id=f"{run_id}:vision:{i}:error",
                               stage="perception", event_type="vision_frame_failed",
                               level="warn",  # type: ignore[arg-type]
                               payload={"frame": i, "error": repr(exc)[:160]}))
            continue
        log_llm_call(sink, run_id, "perception_vision", result, event_id=f"{run_id}:vision_{i}")
        descs = parse_descriptions(result.text, 1)
        if descs and compose_content(descs[0]):
            good_descs.append(descs[0])
            good_kfs.append(kf)

    return descriptions_to_items(good_descs, good_kfs, cfg)
