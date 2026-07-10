"""Non-speech audio: five honest coarse tags, no over-engineered classifier.

From librosa features (onset density, RMS envelope, spectral centroid, silence ratio)
we derive a small set of tags: music_present, applause_or_crowd, impact_or_crash,
mostly_silent, speech_dense. Five tags we can stand behind beat a mislabeled model.

``derive_tags`` is pure; feature extraction is injected via ``feature_fn``.
"""

from __future__ import annotations

from typing import Callable, Optional

from pydantic import BaseModel

from claris.core.perception.config import PerceptionConfig
from claris.core.schema import EvidenceItem, VideoMeta

SOURCE_MODEL = "librosa-heuristic"

_TAG_TEXT = {
    "music_present": "Music is present.",
    "applause_or_crowd": "Applause or crowd noise.",
    "impact_or_crash": "Sharp impacts or crashes.",
    "mostly_silent": "Audio is mostly silent.",
    "speech_dense": "Dense speech activity.",
}


class AudioFeatures(BaseModel):
    duration_s: float
    onset_density: float          # onsets per second
    rms_mean: float
    rms_max: float
    spectral_centroid_mean: float  # Hz
    silence_ratio: float           # fraction of windows below the silence floor


def derive_tags(f: AudioFeatures, cfg: PerceptionConfig) -> list[str]:
    """Map features to coarse tags. Pure and deliberately conservative."""
    tags: list[str] = []
    if f.silence_ratio >= cfg.mostly_silent_ratio:
        tags.append("mostly_silent")
        return tags  # nothing else is trustworthy under mostly-silent audio

    if (
        f.rms_mean > cfg.audio_silence_rms
        and f.spectral_centroid_mean >= cfg.music_centroid_hz
        and f.onset_density < cfg.onset_dense_per_s
    ):
        tags.append("music_present")
    if f.onset_density >= cfg.onset_dense_per_s and f.rms_max > 4 * cfg.audio_silence_rms:
        tags.append("impact_or_crash")
    if cfg.speech_dense_onset_per_s <= f.onset_density < cfg.onset_dense_per_s:
        tags.append("speech_dense")
    if f.rms_mean > cfg.audio_silence_rms and f.spectral_centroid_mean < 1200.0:
        tags.append("applause_or_crowd")
    return tags


def tags_to_items(
    tags: list[str], duration_s: float, source_model: str = SOURCE_MODEL
) -> list[EvidenceItem]:
    """One audio_event EvidenceItem per active tag, spanning the clip. Pure."""
    items: list[EvidenceItem] = []
    for i, tag in enumerate(tags, start=1):
        items.append(
            EvidenceItem(
                id=f"audio_{i:03d}",
                kind="audio_event",
                t_start=0.0,
                t_end=duration_s,
                content=_TAG_TEXT.get(tag, tag),
                confidence=0.7,
                source_model=source_model,
            )
        )
    return items


def _librosa_features(audio_path: str, cfg: PerceptionConfig) -> AudioFeatures:  # pragma: no cover
    import librosa  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    y, sr = librosa.load(audio_path, sr=16000, mono=True)
    duration = float(len(y) / sr) if sr else 0.0
    rms = librosa.feature.rms(y=y)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    onsets = librosa.onset.onset_detect(y=y, sr=sr, units="time")
    silence_ratio = float((rms < cfg.audio_silence_rms).mean()) if rms.size else 1.0
    return AudioFeatures(
        duration_s=duration,
        onset_density=float(len(onsets) / duration) if duration else 0.0,
        rms_mean=float(np.mean(rms)) if rms.size else 0.0,
        rms_max=float(np.max(rms)) if rms.size else 0.0,
        spectral_centroid_mean=float(np.mean(centroid)) if centroid.size else 0.0,
        silence_ratio=silence_ratio,
    )


def analyze_audio(
    audio_path: str,
    meta: VideoMeta,
    cfg: Optional[PerceptionConfig] = None,
    *,
    feature_fn: Optional[Callable[[str, PerceptionConfig], AudioFeatures]] = None,
) -> list[EvidenceItem]:
    """Extract audio features and emit coarse audio_event items.

    ``feature_fn`` is injected in tests. Returns [] when the clip has no audio track.
    """
    cfg = cfg or PerceptionConfig()
    if not meta.has_audio:
        return []
    fn = feature_fn or _librosa_features
    features = fn(audio_path, cfg)
    return tags_to_items(derive_tags(features, cfg), meta.duration_s)
