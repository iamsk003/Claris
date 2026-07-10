"""Tunable knobs for the perception stack. One place, no magic numbers scattered."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PerceptionConfig:
    # Duration window (seconds). Outside it we warn, we do not fail.
    min_duration_s: float = 30.0
    max_duration_s: float = 120.0

    # Keyframe sampling.
    max_keyframes: int = 8
    # PySceneDetect ContentDetector threshold. 27 (its default) over-segments busy/edited
    # clips into many spurious shots; 40 is less trigger-happy and keeps real cuts.
    scene_threshold: float = 40.0
    phash_dedup_distance: int = 6            # hamming distance below which frames are dupes

    # OCR filtering.
    ocr_min_confidence: float = 0.5
    ocr_min_area_frac: float = 0.0008        # drop sub-pixel noise boxes

    # Audio event thresholds (coarse, honest tags only).
    audio_silence_rms: float = 0.01          # below this RMS a window counts as silent
    mostly_silent_ratio: float = 0.6         # silent fraction above this => mostly_silent
    music_centroid_hz: float = 2200.0        # steady mid centroid + low silence => music
    onset_dense_per_s: float = 3.0           # onsets/sec above this => impact_or_crash
    speech_dense_onset_per_s: float = 1.5

    # Vision (Gemma 3 VLM via Fireworks).
    vision_model: str = "accounts/fireworks/models/gemma-3-vlm"
    vision_temperature: float = 0.2
    vision_confidence: float = 0.8           # VLM returns no score; assign a calibrated default
    # Headroom so a reasoning VLM (kimi/qwen fallbacks) can finish and emit the JSON even if
    # reasoning is not fully suppressed; without it the answer gets truncated -> "no detail".
    vision_max_tokens: int = 2048

    # ASR (faster-whisper, in-container, always).
    whisper_model: str = "base"

    # Timeouts (seconds) — nothing in perception runs unbounded.
    ffprobe_timeout_s: float = 30.0
    ffmpeg_timeout_s: float = 60.0
    vision_timeout_s: float = 90.0

    seed: int = 1337
