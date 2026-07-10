"""Perception unit tests. Zero network, zero real video: every heavy stage is mocked."""

from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path

from claris.core.perception.audio_events import (
    AudioFeatures,
    analyze_audio,
    derive_tags,
    tags_to_items,
)
from claris.core.perception.config import PerceptionConfig
from claris.core.perception.ledger import (
    assemble_ledger,
    assign_ids,
    build_ledger,
    compute_modality_flags,
    interval_union_coverage,
)
from claris.core.perception.ocr import OcrBox, dedup_ocr, filter_boxes, normalize_text, run_ocr
from claris.core.perception.probe import duration_warning, parse_ffprobe, parse_fraction
from claris.core.perception.shots import (
    Keyframe,
    classify_motion,
    dedup_by_phash,
    distribute_cap,
    motion_items,
)
from claris.core.perception.speech import SpeechSegment, build_speech_items, segments_to_items
from claris.core.perception.vision import (
    build_prompt,
    compose_content,
    describe_keyframes,
    parse_descriptions,
)
from claris.core.schema import EvidenceItem, Task, VideoMeta, estimate_tokens
from tests.fakes import FakeVisionProvider

CFG = PerceptionConfig()

# 1x1 transparent PNG, so vision's _load_images has real bytes to read.
_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)


def _meta(duration=60.0, has_audio=True):
    return VideoMeta(video_sha256="f" * 64, duration_s=duration, fps=30.0, has_audio=has_audio)


# --------------------------------------------------------------------------- probe


def test_parse_fraction():
    assert parse_fraction("30000/1001") == 30000 / 1001
    assert parse_fraction("25") == 25.0
    assert parse_fraction("1/0") is None
    assert parse_fraction(None) is None


def test_parse_ffprobe_and_warning():
    payload = {
        "format": {"duration": "47.5", "format_name": "mov,mp4,m4a"},
        "streams": [
            {"codec_type": "video", "codec_name": "h264", "width": 1920, "height": 1080,
             "r_frame_rate": "30/1"},
            {"codec_type": "audio", "codec_name": "aac"},
        ],
    }
    fields = parse_ffprobe(payload)
    assert fields["duration_s"] == 47.5
    assert fields["fps"] == 30.0
    assert fields["width"] == 1920 and fields["has_audio"] is True
    assert fields["container"] == "mov"
    assert duration_warning(47.5, CFG) is None
    assert "under" in duration_warning(10.0, CFG)
    assert "exceeds" in duration_warning(200.0, CFG)


def test_parse_ffprobe_no_audio():
    fields = parse_ffprobe({"format": {"duration": "30"}, "streams": [{"codec_type": "video"}]})
    assert fields["has_audio"] is False


# --------------------------------------------------------------------------- shots


def test_distribute_cap_proportional():
    alloc = distribute_cap([40.0, 10.0, 5.0], 8)
    assert sum(alloc) == 8
    assert alloc[0] > alloc[1] >= alloc[2]  # longest shot gets the most frames


def test_distribute_cap_edges():
    assert distribute_cap([], 8) == []
    assert distribute_cap([1.0, 2.0], 0) == [0, 0]
    assert sum(distribute_cap([0.0, 0.0], 4)) == 4  # zero-duration falls back to even


def test_dedup_by_phash():
    # 0 and 1 differ by 1 bit (dupe); 2 differs a lot (kept).
    kept = dedup_by_phash([0b0000, 0b0001, 0b1111_1111], threshold=6)
    assert kept == [0, 2]


def test_classify_motion():
    assert classify_motion([0.001, 0.001], 0, 5.0) == "static"
    assert classify_motion([0.05, 0.05, 0.05], 0, 5.0) == "moving"
    assert classify_motion([0.0, 0.3, 0.0, 0.3], 0, 5.0) == "unsteady"
    assert classify_motion([0.05], 5, 5.0) == "cut-heavy"


def test_motion_items_describe_activity_not_camera():
    items = motion_items([(0.0, 3.0, "moving", [0.05]), (3.0, 6.0, "static", [0.001])])
    assert len(items) == 2
    for it in items:
        assert it.source_model and it.confidence is not None
        assert it.kind.value == "motion"
        # Must NOT assert camera operation (pan/handheld) — that was fabricated evidence.
        assert "camera" not in it.content and "pan" not in it.content


# --------------------------------------------------------------------------- speech


def test_segments_to_items_skips_empty():
    segs = [SpeechSegment(0.0, 1.0, "hello", 0.9), SpeechSegment(1.0, 2.0, "   ", 0.9)]
    items = segments_to_items(segs)
    assert len(items) == 1 and items[0].content == "hello"


def test_build_speech_items_drops_whisper_hallucinations():
    # Near-zero-confidence garbage / punctuation-only "speech" (ASMR, music, silence) is
    # dropped, so the clip resolves to silent rather than emitting "~!"/"♪" noise.
    m = _meta()
    garbage = [SpeechSegment(0.0, 1.9, "~!", 0.0), SpeechSegment(1.9, 3.3, "♪", 0.1)]
    items, is_silent = build_speech_items(garbage, m)
    assert is_silent and "No speech detected" in items[0].content
    # A genuine, confident utterance still survives.
    real, sil = build_speech_items([SpeechSegment(0.0, 1.0, "hello there", 0.8)], m)
    assert not sil and real[0].content == "hello there"


def test_build_speech_items_three_states():
    m = _meta()
    spoken, silent = build_speech_items([SpeechSegment(0.0, 1.0, "hi", 0.9)], m)
    assert not silent and spoken[0].content == "hi"

    none_items, is_silent = build_speech_items([], m)
    assert is_silent and "No speech" in none_items[0].content

    no_audio_items, sil2 = build_speech_items([], _meta(has_audio=False))
    assert not sil2 and "no audio track" in no_audio_items[0].content


# --------------------------------------------------------------------------- ocr


def test_filter_boxes():
    boxes = [
        OcrBox(text="KEEP", confidence=0.9, area_frac=0.01),
        OcrBox(text="low conf", confidence=0.2, area_frac=0.01),
        OcrBox(text="tiny", confidence=0.9, area_frac=0.00001),
        OcrBox(text="  ", confidence=0.9, area_frac=0.01),
    ]
    kept = filter_boxes(boxes, CFG)
    assert [b.text for b in kept] == ["KEEP"]


def test_dedup_ocr_keeps_earliest_and_best():
    frame_boxes = [
        (5.0, OcrBox(text="Login", confidence=0.8, area_frac=0.02)),
        (1.0, OcrBox(text="login", confidence=0.95, area_frac=0.02)),
        (2.0, OcrBox(text="Submit", confidence=0.9, area_frac=0.02)),
    ]
    items = dedup_ocr(frame_boxes)
    assert len(items) == 2
    login = next(i for i in items if normalize_text(i.content) == "login")
    assert login.t_start == 1.0 and login.confidence == 0.95


def test_run_ocr_injected_engine():
    kfs = [
        Keyframe(frame_index=0, t_mid=1.0, shot_index=0, sharpness=1.0, phash=0, image_path="a"),
        Keyframe(frame_index=1, t_mid=2.0, shot_index=1, sharpness=1.0, phash=1, image_path="b"),
    ]
    engine = lambda path: [OcrBox(text=f"text_{path}", confidence=0.9, area_frac=0.02)]
    items, per_frame = run_ocr(kfs, CFG, ocr_fn=engine)
    assert len(items) == 2
    assert per_frame[1.0] == ["text_a"]


# --------------------------------------------------------------------------- audio


def test_derive_tags_mostly_silent_short_circuits():
    f = AudioFeatures(duration_s=60, onset_density=5, rms_mean=0.2, rms_max=0.5,
                      spectral_centroid_mean=3000, silence_ratio=0.9)
    assert derive_tags(f, CFG) == ["mostly_silent"]


def test_derive_tags_music_and_impact():
    music = AudioFeatures(duration_s=60, onset_density=0.5, rms_mean=0.1, rms_max=0.3,
                          spectral_centroid_mean=2500, silence_ratio=0.1)
    assert "music_present" in derive_tags(music, CFG)
    impact = AudioFeatures(duration_s=60, onset_density=5, rms_mean=0.1, rms_max=0.5,
                           spectral_centroid_mean=1500, silence_ratio=0.1)
    assert "impact_or_crash" in derive_tags(impact, CFG)


def test_analyze_audio_no_audio_returns_empty():
    assert analyze_audio("x", _meta(has_audio=False), CFG) == []


def test_tags_to_items_span_clip():
    items = tags_to_items(["music_present"], 60.0)
    assert items[0].t_start == 0.0 and items[0].t_end == 60.0
    assert items[0].confidence is not None and items[0].source_model


# --------------------------------------------------------------------------- vision


def test_parse_descriptions_valid_fenced_and_garbage():
    good = '[{"setting":"kitchen","subjects":["cook"],"actions":["searing"],'\
           '"objects":["pan"],"mood":"calm","notable_detail":"steam"}]'
    descs = parse_descriptions(good, 1)
    assert descs and descs[0].setting == "kitchen"

    fenced = "```json\n" + good + "\n```"
    assert parse_descriptions(fenced, 1)[0].subjects == ["cook"]

    assert parse_descriptions("totally not json", 1) is None


def test_parse_descriptions_pads_to_expected():
    descs = parse_descriptions('[{"setting":"a"}]', 3)
    assert len(descs) == 3


def test_compose_content_grounded():
    from claris.core.perception.vision import FrameDescription

    c = compose_content(FrameDescription(setting="office", subjects=["dev"], objects=["laptop"]))
    assert "office" in c and "dev" in c and c.endswith(".")


def test_build_prompt_includes_ocr():
    kf = Keyframe(frame_index=0, t_mid=3.0, shot_index=0, sharpness=1.0, phash=0, image_path="a")
    prompt = build_prompt([kf], {3.0: ["ERROR 429"]})
    assert "3.0s" in prompt and "ERROR 429" in prompt


def test_describe_keyframes_per_frame_skips_failures():
    # One VLM call per keyframe; a frame that fails to parse is skipped, not placeholdered.
    kfs = [
        Keyframe(frame_index=0, t_mid=1.0, shot_index=0, sharpness=1.0, phash=0, image_path="a"),
        Keyframe(frame_index=1, t_mid=2.0, shot_index=1, sharpness=1.0, phash=1, image_path="b"),
    ]
    valid = '{"setting":"lab","subjects":[],"actions":[],"objects":[],"mood":"","notable_detail":""}'
    provider = FakeVisionProvider(lambda prompt, n, attempt: "broken" if "1.0s" in prompt else valid)
    items = asyncio.run(describe_keyframes(kfs, {}, provider, CFG, images=[_PNG, _PNG]))
    assert len(provider.calls) == 2                 # one call per keyframe, no batching
    assert len(items) == 1 and "lab" in items[0].content  # only the parseable frame survives


# --------------------------------------------------------------------------- ledger


def test_interval_union_coverage():
    items = [
        EvidenceItem(id="x", kind="audio_event", t_start=0.0, t_end=30.0,
                     content="a", confidence=0.7, source_model="m"),
        EvidenceItem(id="y", kind="motion", t_start=20.0, t_end=55.0,
                     content="b", confidence=0.7, source_model="m"),
    ]
    # union [0,55] over 60 => 0.9167
    assert interval_union_coverage(items, 60.0) > 0.9
    assert interval_union_coverage([], 60.0) == 0.0


def test_assign_ids_time_order():
    items = [
        EvidenceItem(id="tmp2", kind="visual", t_start=5.0, t_end=5.0, content="b",
                     confidence=0.8, source_model="m"),
        EvidenceItem(id="tmp1", kind="speech", t_start=1.0, t_end=2.0, content="a",
                     confidence=0.8, source_model="m"),
    ]
    out = assign_ids(items)
    assert [i.id for i in out] == ["E001", "E002"]
    assert out[0].content == "a"  # earliest first


def _realistic_items(duration=60.0):
    items: list[EvidenceItem] = []
    # motion spanning contiguous shots => full coverage
    for i, (a, b) in enumerate([(0, 15), (15, 32), (32, 48), (48, 60)]):
        items.append(EvidenceItem(id=f"m{i}", kind="motion", t_start=float(a), t_end=float(b),
                                  content=f"Shot {i+1}: pan camera motion.", confidence=0.7,
                                  source_model="motion-heuristic"))
    for i in range(12):
        items.append(EvidenceItem(id=f"s{i}", kind="speech", t_start=i * 5.0, t_end=i * 5.0 + 3,
                                  content="A spoken utterance of moderate length about the scene.",
                                  confidence=0.88, source_model="faster-whisper"))
    for i in range(8):
        items.append(EvidenceItem(id=f"v{i}", kind="visual", t_start=i * 7.0, t_end=i * 7.0,
                                  content="A person at a workstation. Subjects: developer. "
                                          "Actions: typing. Objects: laptop, mug. Notable: green check.",
                                  confidence=0.8, source_model="gemma-3-vlm"))
    for i, txt in enumerate(["claris-agent --replay", "INFO 12 done", "Login", "Submit", "v2.1", "429"]):
        items.append(EvidenceItem(id=f"o{i}", kind="ocr", t_start=float(i), t_end=float(i),
                                  content=txt, confidence=0.93, source_model="paddleocr"))
    items.append(EvidenceItem(id="a0", kind="audio_event", t_start=0.0, t_end=duration,
                              content="Dense speech activity.", confidence=0.7,
                              source_model="librosa-heuristic"))
    return items


def test_assemble_ledger_coverage_and_every_item_annotated():
    task = Task(task_id="clip_x", video_path="clip_x.mp4")
    ledger = assemble_ledger(task, _meta(), _realistic_items(), ["gemma-3-vlm", "faster-whisper"])
    assert ledger.coverage >= 0.85
    assert ledger.modality_flags.has_speech and ledger.modality_flags.has_ocr
    assert ledger.modality_flags.has_visual and ledger.modality_flags.has_motion
    for it in ledger.items:
        assert it.source_model and it.confidence is not None
        assert it.id.startswith("E")


def test_60s_ledger_serializes_under_2500_tokens():
    task = Task(task_id="clip_x", video_path="clip_x.mp4")
    ledger = assemble_ledger(task, _meta(), _realistic_items(), ["gemma-3-vlm"])
    block = ledger.to_prompt_block()
    tokens = estimate_tokens(block)
    assert tokens < 2500, f"ledger prompt block is {tokens} tokens"


def test_compute_modality_flags_silent():
    m = _meta()
    speech_only = [EvidenceItem(id="s", kind="speech", t_start=0.0, t_end=60.0,
                                content="No speech detected in this clip.", confidence=0.9,
                                source_model="faster-whisper")]
    flags = compute_modality_flags(speech_only, m, is_silent=True)
    assert flags.is_silent and not flags.has_speech


def test_build_ledger_end_to_end_offline(tmp_path: Path):
    # Real tiny PNGs so vision's image loader has bytes; everything else injected.
    kfs = []
    for i in range(3):
        p = tmp_path / f"kf_{i}.png"
        p.write_bytes(_PNG)
        kfs.append(Keyframe(frame_index=i, t_mid=i * 20.0 + 5, shot_index=i,
                            sharpness=1.0, phash=i, image_path=str(p)))
    motion = motion_items([(0.0, 20.0, "pan", [0.05]), (20.0, 40.0, "static", [0.001]),
                           (40.0, 60.0, "handheld", [0.2])])

    valid = json.dumps([
        {"setting": "office", "subjects": ["dev"], "actions": ["typing"],
         "objects": ["laptop"], "mood": "focused", "notable_detail": "green check"}
    ] * 3)

    ledger = asyncio.run(build_ledger(
        Task(task_id="clip_e2e", video_path="clip.mp4"),
        CFG,
        vision_provider=FakeVisionProvider(lambda prompt, n, attempt: valid),
        video_path="clip.mp4",
        probe_fn=lambda p, c: (_meta(60.0, True), None),
        extract_audio_fn=lambda p, c: str(tmp_path / "a.wav"),
        keyframe_fn=lambda p, c: (kfs, motion),
        ocr_fn=lambda path: [OcrBox(text="ERROR 429", confidence=0.9, area_frac=0.02)],
        transcribe_fn=lambda p, c: [SpeechSegment(2.0, 5.0, "hello world", 0.9)],
        audio_feature_fn=lambda p, c: AudioFeatures(
            duration_s=60, onset_density=2.0, rms_mean=0.1, rms_max=0.3,
            spectral_centroid_mean=1500, silence_ratio=0.1),
    ))

    assert ledger.task_id == "clip_e2e"
    assert ledger.coverage >= 0.85
    assert ledger.modality_flags.has_visual and ledger.modality_flags.has_ocr
    assert ledger.modality_flags.has_speech and ledger.modality_flags.has_motion
    assert all(i.id.startswith("E") for i in ledger.items)
    assert all(i.source_model and i.confidence is not None for i in ledger.items)
    # OCR string survived into the ledger, grounding humorous_tech downstream.
    assert any("429" in i.content for i in ledger.items if i.kind.value == "ocr")
