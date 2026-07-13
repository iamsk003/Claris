"""End-to-end pipeline tests for CLARIS v2 — one reasoning call, fully offline."""

from __future__ import annotations

import asyncio
import json

from claris.core.observability import ListSink
from claris.core.perception.shots import Keyframe
from claris.core.pipeline import Providers, run_from_ledger, run_pipeline
from claris.core.schema import ALL_STYLES, EvidenceItem, EvidenceKind, Task, VideoMeta
from eval.harness import load_golden_ledgers
from tests.fakes import FakeVisionProvider


def _task(ledger):
    return Task(task_id=ledger.task_id, video_path="x.mp4", styles=ALL_STYLES)


def _reply(prompt, n_images, attempt) -> str:
    return json.dumps({
        "summary": "A short clip.",
        "events": [{"time": 0.0, "description": "opening"}],
        "captions": {
            "canonical": "A subject moves through a short clip.",
            "formal": "A subject is shown moving through a short clip.",
            "sarcastic": "Riveting footage of a subject moving. Truly.",
            "tech_humor": "Rendering the subject at a stable 24fps, no dropped frames.",
            "everyday_humor": "Just a subject out here living its best clip life.",
        },
        "grounded": True,
        "modalities": {"speech_used": True, "ocr_used": False, "audio_used": True},
    })


def _providers() -> Providers:
    return Providers(reasoning_provider=FakeVisionProvider(_reply))


def test_pipeline_produces_four_grounded_captions():
    ledger = load_golden_ledgers()[0]
    sink = ListSink()
    result = asyncio.run(run_from_ledger(ledger, _task(ledger), _providers(), sink=sink))
    assert set(result.captions) == set(ALL_STYLES)
    assert result.degraded is False
    for cap in result.captions.values():
        assert cap.text and cap.evidence_ids  # grounded on the ledger's items
    types = {e.event_type for e in sink.events}
    assert {"reasoning_start", "reasoning_done"} <= types


def test_pipeline_absorbs_retired_v1_kwargs():
    # Callers may still pass old knobs; they must be ignored, not raise.
    ledger = load_golden_ledgers()[0]
    result = asyncio.run(run_from_ledger(
        ledger, _task(ledger), _providers(),
        use_gates=True, gen_config=object(), ver_config=object(), registry=object(),
    ))
    assert set(result.captions) == set(ALL_STYLES)


def test_missing_reasoning_provider_degrades_to_template():
    ledger = load_golden_ledgers()[0]
    result = asyncio.run(run_from_ledger(ledger, _task(ledger), Providers()))
    assert set(result.captions) == set(ALL_STYLES)
    assert result.degraded is True  # no provider -> template captions, never absent


def test_run_pipeline_perception_to_reasoning_end_to_end(tmp_path):
    # video -> build_perception (injected fakes, no ffmpeg) -> one reasoning call -> captions.
    img = tmp_path / "frame.jpg"
    img.write_bytes(b"\xff\xd8fakejpeg")
    kf = Keyframe(frame_index=0, t_mid=1.0, shot_index=0, sharpness=1.0, phash=0,
                  image_path=str(img))
    motion = EvidenceItem(id="m1", kind=EvidenceKind.MOTION, t_start=0.0, t_end=2.0,
                          content="steady on-screen movement", confidence=0.6,
                          source_model="motion")

    def probe_fn(path, cfg):
        return VideoMeta(video_sha256="x", duration_s=45.0, has_audio=False), None

    result = asyncio.run(run_pipeline(
        Task(task_id="t1", video_path="x.mp4"), _providers(),
        probe_fn=probe_fn, keyframe_fn=lambda p, c: ([kf], [motion]),
        transcribe_fn=lambda path, cfg: ([], False),
    ))
    assert set(result.captions) == set(ALL_STYLES)
    assert result.degraded is False
    assert all("could not be captioned" not in c.text for c in result.captions.values())


def test_submission_shape():
    ledger = load_golden_ledgers()[0]
    result = asyncio.run(run_from_ledger(ledger, _task(ledger), _providers()))
    sub = result.to_submission()
    assert set(sub["captions"]) == {s.value for s in ALL_STYLES}
    assert all(isinstance(v, str) and v for v in sub["captions"].values())
