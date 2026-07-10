"""ffprobe wrapper: duration, fps, resolution, has_audio, sha256.

The ffprobe call is isolated in ``_run_ffprobe``; everything else is pure and unit
tested against a captured ffprobe payload, so no video or binary is needed in tests.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Optional

from claris.core.perception.config import PerceptionConfig
from claris.core.schema import VideoMeta


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_fraction(value: Optional[str]) -> Optional[float]:
    """Parse an ffprobe fraction like '30000/1001' into a float."""
    if not value:
        return None
    try:
        if "/" in value:
            num, den = value.split("/", 1)
            den_f = float(den)
            return float(num) / den_f if den_f else None
        return float(value)
    except (ValueError, ZeroDivisionError):
        return None


def parse_ffprobe(data: dict[str, Any]) -> dict[str, Any]:
    """Turn a raw ffprobe JSON payload into VideoMeta kwargs. Pure."""
    streams = data.get("streams", [])
    fmt = data.get("format", {})
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)

    duration = None
    for source in (fmt.get("duration"), (video or {}).get("duration")):
        if source is not None:
            try:
                duration = float(source)
                break
            except (TypeError, ValueError):
                continue

    fps = parse_fraction((video or {}).get("r_frame_rate")) if video else None
    width = (video or {}).get("width") if video else None
    height = (video or {}).get("height") if video else None
    container = (fmt.get("format_name") or "").split(",")[0] or None

    return {
        "duration_s": duration if duration is not None else 0.0,
        "fps": fps,
        "width": width,
        "height": height,
        "has_audio": audio is not None,
        "container": container,
        "video_codec": (video or {}).get("codec_name") if video else None,
        "audio_codec": (audio or {}).get("codec_name") if audio else None,
    }


def duration_warning(duration_s: float, cfg: PerceptionConfig) -> Optional[str]:
    """Return a warning string if duration is outside the window, else None."""
    if duration_s < cfg.min_duration_s:
        return f"duration {duration_s:.1f}s is under the {cfg.min_duration_s:.0f}s minimum"
    if duration_s > cfg.max_duration_s:
        return f"duration {duration_s:.1f}s exceeds the {cfg.max_duration_s:.0f}s maximum"
    return None


def _run_ffprobe(video_path: str | Path, timeout_s: float) -> dict[str, Any]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_format",
        "-show_streams",
        "-print_format",
        "json",
        str(video_path),
    ]
    proc = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout_s, check=True
    )
    return json.loads(proc.stdout)


def probe(
    video_path: str | Path,
    cfg: Optional[PerceptionConfig] = None,
    *,
    ffprobe_fn=None,
) -> tuple[VideoMeta, Optional[str]]:
    """Probe a video file into VideoMeta plus an optional duration warning.

    ``ffprobe_fn`` is injectable for tests; it defaults to the real ffprobe call.
    """
    cfg = cfg or PerceptionConfig()
    ffprobe_fn = ffprobe_fn or (lambda p: _run_ffprobe(p, cfg.ffprobe_timeout_s))
    raw = ffprobe_fn(video_path)
    fields = parse_ffprobe(raw)
    meta = VideoMeta(video_sha256=sha256_file(video_path), **fields)
    return meta, duration_warning(meta.duration_s, cfg)
