"""CLARIS v2 reasoning: structured evidence in, one JSON with four captions out."""

from __future__ import annotations

import json

import pytest

from claris.core.perception.bundle import PerceptionBundle
from claris.core.perception.shots import Keyframe
from claris.core.reasoning import (
    build_evidence_json,
    parse_reasoning,
    reason_over_clip,
    reasoning_to_result,
)
from claris.core.schema import (
    EvidenceItem,
    EvidenceKind,
    EvidenceLedger,
    ProviderTier,
    StyleName,
    Task,
    VideoMeta,
)
from tests.fakes import FakeVisionProvider


def _ledger() -> EvidenceLedger:
    items = (
        EvidenceItem(id="E001", kind=EvidenceKind.MOTION, t_start=0.0, t_end=3.0,
                     content="steady on-screen movement", confidence=0.6, source_model="motion"),
        EvidenceItem(id="E002", kind=EvidenceKind.SPEECH, t_start=1.0, t_end=4.0,
                     content="today we bake bread", confidence=0.9, source_model="whisper"),
        EvidenceItem(id="E003", kind=EvidenceKind.OCR, t_start=2.0, t_end=2.5,
                     content="FRESH", confidence=0.8, source_model="paddleocr"),
    )
    return EvidenceLedger(
        ledger_id="led_t1", task_id="t1", video_sha256="abc",
        video_meta=VideoMeta(video_sha256="abc", duration_s=10.0), items=items,
    )


def _bundle() -> PerceptionBundle:
    kf = Keyframe(frame_index=0, t_mid=1.0, shot_index=0, sharpness=1.0, phash=0,
                  image_path="/nonexistent.jpg")
    return PerceptionBundle(ledger=_ledger(), keyframes=(kf, kf))


def _reply(**overrides) -> str:
    body = {
        "summary": "A person bakes bread.",
        "events": [{"time": 0.0, "description": "intro"}],
        "captions": {
            "canonical": "A person bakes bread while narrating.",
            "formal": "A person prepares bread while describing the process.",
            "sarcastic": "Ah yes, groundbreaking bread content.",
            "tech_humor": "Compiling dough; build succeeded, tastes shipped.",
            "everyday_humor": "Carbs incoming, and honestly no regrets.",
        },
        "grounded": True,
        "modalities": {"speech_used": True, "ocr_used": True, "audio_used": False},
    }
    body.update(overrides)
    return json.dumps(body)


def test_build_evidence_json_groups_by_kind():
    ev = build_evidence_json(_ledger())
    assert ev["transcript"] == [{"time": 1.0, "text": "today we bake bread"}]
    assert ev["ocr"] == [{"time": 2.0, "text": "FRESH"}]
    assert ev["motion"] == [{"time": 0.0, "event": "steady on-screen movement"}]
    assert ev["audio_events"] == []


def test_parse_reasoning_handles_fenced_json():
    data = parse_reasoning("```json\n" + _reply() + "\n```")
    assert data is not None and data["captions"]["formal"].startswith("A person")


@pytest.mark.asyncio
async def test_reason_over_clip_emits_four_grounded_captions():
    provider = FakeVisionProvider(lambda prompt, n, attempt: _reply())
    result = await reason_over_clip(
        _bundle(), Task(task_id="t1", video_path="x.mp4"), provider,
        image_loader=lambda b: [b"img"] * len(b.keyframes),
    )
    assert set(result.captions) == set(StyleName)
    assert result.degraded is False
    assert provider.calls == [2]  # exactly one multimodal call, two frames
    tech = result.captions[StyleName.HUMOROUS_TECH]
    assert "build succeeded" in tech.text
    assert tech.evidence_ids == ("E001", "E002", "E003")


def test_ungrounded_flags_every_caption_degraded():
    result = reasoning_to_result(
        Task(task_id="t1", video_path="x.mp4"), _ledger(),
        json.loads(_reply(grounded=False)),
        tier=ProviderTier.FIREWORKS_NON_GEMMA, model="kimi", run_id="r1",
    )
    assert result.degraded is True
    assert all(c.degraded_ungrounded for c in result.captions.values())


def test_unparseable_reply_falls_back_to_template():
    result = reasoning_to_result(
        Task(task_id="t1", video_path="x.mp4"), _ledger(), None,
        tier=ProviderTier.TEMPLATE, model="unavailable", run_id="r1",
    )
    assert result.degraded is True
    assert all(c.provider_tier == ProviderTier.TEMPLATE for c in result.captions.values())
    assert all(c.text for c in result.captions.values())
