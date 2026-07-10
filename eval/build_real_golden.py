"""Build the real golden ledgers from real clips (item 1).

Runs the actual perception stack — content-aware shots (OpenCV + PySceneDetect), ASR
(faster-whisper), OCR (PaddleOCR), audio events (librosa), and Gemma 3 VLM keyframe
understanding via Fireworks — over each clip, writes the ledger to eval/golden/, and
reports per-clip coverage plus any timeline dead zones.

    export FIREWORKS_API_KEY=fw_...
    python -m eval.build_real_golden eval/clips/*.mp4

Requires the heavy perception deps (uv sync) and a Fireworks key. This is the runner the
eval/clips/README points at; it is intentionally not exercised by the offline test suite.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from claris.core.perception import PerceptionConfig, build_ledger
from claris.core.schema import EvidenceLedger, Task

GOLDEN_DIR = Path(__file__).parent / "golden"
COVERAGE_BAR = 0.85


def timeline_gaps(ledger: EvidenceLedger, min_gap_s: float = 1.0) -> list[tuple[float, float]]:
    """Uncovered spans longer than ``min_gap_s`` — the dead zones sampling may have left."""
    duration = ledger.video_meta.duration_s
    spans = sorted((max(0.0, it.t_start), min(duration, max(it.t_start, it.t_end)))
                   for it in ledger.items)
    gaps: list[tuple[float, float]] = []
    cursor = 0.0
    for start, end in spans:
        if start > cursor + min_gap_s:
            gaps.append((round(cursor, 1), round(start, 1)))
        cursor = max(cursor, end)
    if duration > cursor + min_gap_s:
        gaps.append((round(cursor, 1), round(duration, 1)))
    return gaps


async def _build_one(clip: Path, index: int) -> EvidenceLedger:
    from claris.core.providers.fireworks import FireworksVisionProvider  # noqa: PLC0415

    task = Task(task_id=f"clip_{index:02d}_{clip.stem}", video_path=str(clip))
    ledger = await build_ledger(
        task, PerceptionConfig(), vision_provider=FireworksVisionProvider(),
        video_path=str(clip),
    )
    out = GOLDEN_DIR / f"ledger_{index:02d}_{clip.stem}.json"
    out.write_text(json.dumps(ledger.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n")
    return ledger


def main(argv: list[str]) -> int:
    clips = [Path(p) for p in argv]
    if not clips:
        print("usage: python -m eval.build_real_golden <clip> [clip ...]")
        return 2

    below = []
    for i, clip in enumerate(clips, start=1):
        ledger = asyncio.run(_build_one(clip, i))
        gaps = timeline_gaps(ledger)
        flag = "" if ledger.coverage >= COVERAGE_BAR else "  <-- below 0.85"
        print(f"{clip.name}: coverage {ledger.coverage:.3f}, {len(ledger.items)} items{flag}")
        if gaps:
            print(f"    dead zones (uncovered > 1s): {gaps}")
        if ledger.coverage < COVERAGE_BAR:
            below.append((clip.name, ledger.coverage, gaps))

    if below:
        print("\nSome clips are below the 0.85 coverage bar. Do NOT lower the threshold; "
              "the dead zones above show whether shot sampling is skipping timeline regions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
