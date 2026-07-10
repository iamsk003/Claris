"""Regenerate the golden ledgers through the REAL perception assembly path.

These fixtures are produced by ``assemble_ledger`` (the same code perception runs), so
they carry real E-prefixed time-ordered IDs, computed ``coverage`` and ``modality_flags``,
and serialize exactly as ``to_prompt_block`` will render them. They are still authored
scenarios, not real clips — real clips arrive before Prompt 3. Re-run:

    python -m eval.golden._build_fixtures
"""

from __future__ import annotations

import json
from pathlib import Path

from claris.core.perception.ledger import assemble_ledger
from claris.core.schema import EvidenceItem, Task, VideoMeta

OUT_DIR = Path(__file__).parent


def _item(kind, t0, t1, content, conf, model) -> EvidenceItem:
    # id is a placeholder; assemble_ledger reassigns E001.. in time order.
    return EvidenceItem(
        id="tmp", kind=kind, t_start=t0, t_end=t1, content=content,
        confidence=conf, source_model=model,
    )


VLM, ASR, PANNS, OCR = "gemma-3-vlm", "whisper-large-v3", "panns-audio-tagger", "paddle-ocr"


def cooking():
    meta = VideoMeta(video_sha256="0" * 63 + "1", duration_s=47.0, fps=30.0, width=1920,
                     height=1080, has_audio=True, container="mp4", video_codec="h264",
                     audio_codec="aac")
    items = [
        _item("visual", 0.0, 3.5, "A person in a blue apron stands at a stovetop holding a cast-iron skillet.", 0.94, VLM),
        _item("speech", 1.2, 5.0, "Okay, the pan is screaming hot, so we're going to lay the steak down away from us.", 0.90, ASR),
        _item("audio_event", 5.0, 9.0, "Loud sizzling sound.", 0.88, PANNS),
        _item("visual", 5.0, 12.0, "A thick steak is placed in the skillet; steam and smoke rise from the pan.", 0.91, VLM),
        _item("motion", 18.0, 20.0, "The cook flips the steak with tongs.", 0.82, VLM),
        _item("speech", 22.0, 27.5, "We add a knob of butter, some thyme, and a smashed garlic clove, then baste.", 0.87, ASR),
        _item("visual", 28.0, 38.0, "The cook tilts the pan and spoons foaming butter over the steak repeatedly.", 0.90, VLM),
        _item("visual", 40.0, 47.0, "The finished steak rests on a wooden board and is sliced to show a pink interior.", 0.89, VLM),
    ]
    task = Task(task_id="golden_cooking", video_path="cooking.mp4")
    return "ledger_01_cooking.json", assemble_ledger(task, meta, items, [VLM, ASR, PANNS], is_silent=False)


def coding_demo():
    meta = VideoMeta(video_sha256="0" * 63 + "2", duration_s=63.0, fps=24.0, width=2560,
                     height=1440, has_audio=True, container="mp4", video_codec="h264",
                     audio_codec="aac")
    items = [
        _item("visual", 0.0, 4.0, "A screen-share of a dark-themed terminal next to a code editor.", 0.95, VLM),
        _item("speech", 0.5, 6.0, "So I built a little CLI that batches API calls and retries them with backoff.", 0.92, ASR),
        _item("ocr", 4.0, 10.0, "$ claris-agent --input tasks.json --replay", 0.97, OCR),
        _item("ocr", 10.0, 16.0, "def retry(fn, attempts=5, base=0.5): ...", 0.85, OCR),
        _item("speech", 12.0, 19.0, "If it hits a 429 it waits, jitters, and tries again instead of just dying.", 0.90, ASR),
        _item("motion", 20.0, 22.0, "The presenter runs the command; log lines scroll rapidly in the terminal.", 0.80, VLM),
        _item("ocr", 23.0, 30.0, "[INFO] 12 tasks completed, 0 failed, 3 retried", 0.93, OCR),
        _item("speech", 31.0, 38.0, "Twelve done, zero failed. The three that retried were rate limits, not bugs.", 0.88, ASR),
        _item("visual", 45.0, 63.0, "A green checkmark and a summary panel appear; the presenter leans back smiling.", 0.86, VLM),
    ]
    task = Task(task_id="golden_coding_demo", video_path="coding_demo.mp4")
    return "ledger_02_coding_demo.json", assemble_ledger(task, meta, items, [VLM, ASR, OCR], is_silent=False)


def dog_park():
    meta = VideoMeta(video_sha256="0" * 63 + "3", duration_s=34.0, fps=60.0, width=1280,
                     height=720, has_audio=True, container="mov", video_codec="hevc",
                     audio_codec="aac")
    items = [
        _item("visual", 0.0, 4.0, "A grassy park on a bright day; a golden retriever stands alert near a person.", 0.93, VLM),
        _item("motion", 4.0, 5.0, "A red frisbee is thrown across the frame from left to right.", 0.84, VLM),
        _item("motion", 5.0, 11.0, "The dog sprints after the frisbee, kicking up grass.", 0.90, VLM),
        _item("audio_event", 6.0, 8.0, "Faint background chatter and wind noise.", 0.41, PANNS),
        _item("motion", 11.0, 13.0, "The dog leaps and catches the frisbee mid-air.", 0.79, VLM),
        _item("speech", 13.5, 15.0, "Good boy!", 0.55, ASR),
        _item("visual", 16.0, 24.0, "The dog trots back with the frisbee in its mouth, tail wagging.", 0.88, VLM),
        _item("audio_event", 24.0, 26.0, "A single short bark.", 0.62, PANNS),
    ]
    task = Task(task_id="golden_dog_park", video_path="dog_park.mp4")
    return "ledger_03_dog_park.json", assemble_ledger(task, meta, items, [VLM, ASR, PANNS], is_silent=False)


def main() -> None:
    for builder in (cooking, coding_demo, dog_park):
        name, ledger = builder()
        (OUT_DIR / name).write_text(
            json.dumps(ledger.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n"
        )
        print(f"{name}: {len(ledger.items)} items, coverage {ledger.coverage}, "
              f"modalities {list(ledger.modality_flags.active())}")


if __name__ == "__main__":
    main()
