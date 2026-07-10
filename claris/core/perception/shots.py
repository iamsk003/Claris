"""Content-aware keyframe sampling.

Competitors sample N evenly-spaced frames and miss the shot where the thing happens.
We detect shot boundaries (PySceneDetect ContentDetector), pick the sharpest frame
near each shot's midpoint, deduplicate near-identical frames by perceptual hash, and
distribute a keyframe budget across shots in proportion to their duration.

The heavy work (scenedetect + OpenCV) lives in ``extract_keyframes`` behind lazy
imports; the allocation, motion, and dedup logic is pure and unit tested.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel

from claris.core.perception.config import PerceptionConfig
from claris.core.schema import EvidenceItem


class Keyframe(BaseModel):
    """One selected frame, ready for OCR and the VLM."""

    frame_index: int
    t_mid: float
    shot_index: int
    sharpness: float
    phash: int
    image_path: str


def distribute_cap(shot_durations: list[float], cap: int) -> list[int]:
    """Allocate ``cap`` keyframes across shots proportionally to duration.

    Largest-remainder method: each shot gets floor(share), then leftover slots go to
    the largest fractional remainders. Short shots may receive zero — that is the
    point; we spend the budget where there is the most footage.
    """
    n = len(shot_durations)
    if n == 0 or cap <= 0:
        return [0] * n
    total = sum(shot_durations)
    if total <= 0:
        # Degenerate: spread as evenly as possible.
        base, extra = divmod(cap, n)
        return [base + (1 if i < extra else 0) for i in range(n)]

    exact = [d / total * cap for d in shot_durations]
    alloc = [int(x) for x in exact]
    remaining = cap - sum(alloc)
    order = sorted(range(n), key=lambda i: exact[i] - alloc[i], reverse=True)
    for i in order[:remaining]:
        alloc[i] += 1
    return alloc


def hamming(a: int, b: int) -> int:
    """Hamming distance between two integer perceptual hashes."""
    return bin(a ^ b).count("1")


def dedup_by_phash(hashes: list[int], threshold: int) -> list[int]:
    """Return indices to keep: the first of each near-duplicate cluster. Pure."""
    kept: list[int] = []
    for i, h in enumerate(hashes):
        if all(hamming(h, hashes[k]) > threshold for k in kept):
            kept.append(i)
    return kept


# Frame-difference energy measures how much the FRAME changes, which can come from the
# subject as easily as the camera. So the labels describe on-screen visual activity and
# must NOT assert camera operation (pan/handheld) — that was fabricated evidence.
_MOTION_TEXT = {
    "static": "little on-screen movement",
    "moving": "steady on-screen movement",
    "unsteady": "unsteady, high on-screen movement",
    "cut-heavy": "frequent scene changes",
}


def classify_motion(diff_energies: list[float], cut_count: int, shot_duration: float) -> str:
    """Coarse visual-activity label from frame-difference energy. Pure.

    Describes how much the frame changes, not what caused it: static (little change),
    moving (steady change), unsteady (high-variance change), cut-heavy (many internal cuts).
    """
    if shot_duration > 0 and cut_count / shot_duration > 0.5:
        return "cut-heavy"
    if not diff_energies:
        return "static"
    mean = sum(diff_energies) / len(diff_energies)
    var = sum((d - mean) ** 2 for d in diff_energies) / len(diff_energies)
    if mean < 0.03:
        return "static"
    if var > 0.015:
        return "unsteady"
    return "moving"


def _motion_confidence(diff_energies: list[float]) -> float:
    if not diff_energies:
        return 0.5
    mean = sum(diff_energies) / len(diff_energies)
    return round(min(1.0, 0.5 + mean * 5.0), 3)


def motion_items(
    shots: list[tuple[float, float, str, list[float]]],
    source_model: str = "motion-heuristic",
) -> list[EvidenceItem]:
    """Build motion EvidenceItems from (t_start, t_end, label, diffs) tuples. Pure."""
    items: list[EvidenceItem] = []
    for idx, (t0, t1, label, diffs) in enumerate(shots, start=1):
        items.append(
            EvidenceItem(
                id=f"motion_{idx:03d}",
                kind="motion",
                t_start=t0,
                t_end=t1,
                content=f"Shot {idx}: {_MOTION_TEXT.get(label, label)}.",
                confidence=_motion_confidence(diffs),
                source_model=source_model,
            )
        )
    return items


def extract_keyframes(
    video_path: str,
    cfg: Optional[PerceptionConfig] = None,
    *,
    cache_dir: str = ".claris_cache",
) -> tuple[list[Keyframe], list[EvidenceItem]]:  # pragma: no cover - needs OpenCV + a video
    """Detect shots, select sharp keyframes, dedup, and emit motion evidence.

    Heavy path: imports scenedetect and OpenCV lazily. Returns the kept keyframes and
    one motion EvidenceItem per shot.
    """
    import cv2  # noqa: PLC0415
    import imagehash  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415
    from scenedetect import ContentDetector, SceneManager, open_video  # noqa: PLC0415

    cfg = cfg or PerceptionConfig()
    cache = _ensure_dir(cache_dir)

    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=cfg.scene_threshold))
    scene_manager.detect_scenes(video)
    scenes = scene_manager.get_scene_list()

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    if not scenes:
        total = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps if fps else 0.0
        shot_spans = [(0.0, total)]
    else:
        shot_spans = [(s.get_seconds(), e.get_seconds()) for s, e in scenes]

    durations = [max(0.0, t1 - t0) for t0, t1 in shot_spans]
    alloc = distribute_cap(durations, cfg.max_keyframes)

    raw_frames: list[Keyframe] = []
    hashes: list[int] = []
    shot_meta: list[tuple[float, float, str, list[float]]] = []

    for shot_index, ((t0, t1), n_frames) in enumerate(zip(shot_spans, alloc)):
        diffs = _frame_diffs(cap, fps, t0, t1)
        shot_meta.append((t0, t1, classify_motion(diffs, 0, t1 - t0), diffs))
        if n_frames <= 0:
            continue
        for k in range(n_frames):
            frac = (k + 1) / (n_frames + 1)
            t = t0 + (t1 - t0) * frac
            frame = _read_frame_at(cap, fps, t)
            if frame is None:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            ph = int(str(imagehash.phash(pil)), 16)
            frame_idx = int(t * fps)
            path = f"{cache}/kf_{shot_index:02d}_{k:02d}.png"
            cv2.imwrite(path, frame)
            raw_frames.append(
                Keyframe(
                    frame_index=frame_idx,
                    t_mid=t,
                    shot_index=shot_index,
                    sharpness=sharpness,
                    phash=ph,
                    image_path=path,
                )
            )
            hashes.append(ph)

    cap.release()
    keep = dedup_by_phash(hashes, cfg.phash_dedup_distance)
    keyframes = [raw_frames[i] for i in keep]
    return keyframes, motion_items(shot_meta)


def _ensure_dir(path: str) -> str:
    from pathlib import Path  # noqa: PLC0415

    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def _read_frame_at(cap, fps: float, t: float):  # pragma: no cover
    import cv2  # noqa: PLC0415

    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
    ok, frame = cap.read()
    return frame if ok else None


def _frame_diffs(cap, fps: float, t0: float, t1: float, samples: int = 6):  # pragma: no cover
    import cv2  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    diffs: list[float] = []
    prev = None
    span = max(1e-3, t1 - t0)
    for i in range(samples):
        frame = _read_frame_at(cap, fps, t0 + span * (i / samples))
        if frame is None:
            continue
        small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (64, 64))
        if prev is not None:
            diffs.append(float(np.abs(small.astype("float") - prev).mean()) / 255.0)
        prev = small.astype("float")
    return diffs
