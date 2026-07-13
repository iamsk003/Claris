"""Assemble every signal channel into one chronological EvidenceLedger.

Merges motion, OCR, speech, audio-event and visual items, assigns stable time-ordered
IDs (E001, E002, ...), computes clip ``coverage`` (fraction of duration covered by at
least one item) and ``modality_flags``. The assembly functions are pure and unit
tested; ``build_ledger`` orchestrates the heavy stages behind injectable callables so
the whole path can run offline with fakes.
"""

from __future__ import annotations

import traceback
from typing import Callable, Optional

from claris.core.observability import EventSink, NullSink
from claris.core.perception.audio_events import analyze_audio
from claris.core.perception.bundle import PerceptionBundle
from claris.core.perception.config import PerceptionConfig
from claris.core.perception.shots import Keyframe, extract_keyframes
from claris.core.perception.speech import SpeechSegment, transcribe
from claris.core.perception.vision import describe_keyframes
from claris.core.providers.base import VisionProvider
from claris.core.schema import (
    EvidenceItem,
    EvidenceLedger,
    ModalityFlags,
    RunEvent,
    Task,
    VideoMeta,
)


def interval_union_coverage(items: list[EvidenceItem], duration_s: float) -> float:
    """Fraction of [0, duration] covered by the union of item time ranges. Pure."""
    if duration_s <= 0:
        return 0.0
    spans = sorted(
        (max(0.0, it.t_start), min(duration_s, max(it.t_start, it.t_end))) for it in items
    )
    covered = 0.0
    cur_start: Optional[float] = None
    cur_end = 0.0
    for start, end in spans:
        if end <= start:
            continue
        if cur_start is None:
            cur_start, cur_end = start, end
        elif start <= cur_end:
            cur_end = max(cur_end, end)
        else:
            covered += cur_end - cur_start
            cur_start, cur_end = start, end
    if cur_start is not None:
        covered += cur_end - cur_start
    return round(min(1.0, covered / duration_s), 4)


def compute_modality_flags(
    items: list[EvidenceItem], meta: VideoMeta, is_silent: bool
) -> ModalityFlags:
    """Derive which channels actually carried signal. Pure."""
    kinds = {it.kind.value for it in items}
    return ModalityFlags(
        has_speech="speech" in kinds and meta.has_audio and not is_silent,
        has_ocr="ocr" in kinds,
        has_audio_event="audio_event" in kinds,
        has_visual="visual" in kinds,
        has_motion="motion" in kinds,
        is_silent=is_silent,
    )


def assign_ids(items: list[EvidenceItem]) -> list[EvidenceItem]:
    """Sort chronologically and renumber to E001, E002, ... Pure.

    Ties break by end time then kind, so ordering is deterministic and replayable.
    """
    ordered = sorted(items, key=lambda it: (it.t_start, it.t_end, it.kind.value, it.content))
    return [it.model_copy(update={"id": f"E{i:03d}"}) for i, it in enumerate(ordered, start=1)]


def assemble_ledger(
    task: Task,
    meta: VideoMeta,
    items: list[EvidenceItem],
    perception_models: list[str],
    *,
    is_silent: bool = False,
) -> EvidenceLedger:
    """Build the final immutable EvidenceLedger from all collected items. Pure."""
    numbered = assign_ids(items)
    return EvidenceLedger(
        ledger_id=f"led_{task.task_id}",
        task_id=task.task_id,
        video_sha256=meta.video_sha256,
        video_meta=meta,
        items=tuple(numbered),
        perception_models=tuple(dict.fromkeys(perception_models)),
        coverage=interval_union_coverage(numbered, meta.duration_s),
        modality_flags=compute_modality_flags(numbered, meta, is_silent),
    )


async def _collect_signals(
    task: Task,
    cfg: PerceptionConfig,
    sink: EventSink,
    *,
    video_path: str,
    probe_fn: Optional[Callable],
    extract_audio_fn: Optional[Callable],
    keyframe_fn: Optional[Callable],
    transcribe_fn: Optional[Callable[[str, PerceptionConfig], list[SpeechSegment]]],
    audio_feature_fn: Optional[Callable],
) -> tuple[VideoMeta, list[Keyframe], dict, list[EvidenceItem], bool, Callable]:
    """Run every non-visual perception channel. Shared by build_ledger + build_perception.

    Returns (meta, keyframes, per_frame_ocr, non_visual_items, is_silent, warn_fn). Each
    heavy stage is injectable and failure-isolated: one dead modality never sinks the rest.
    """
    probe_fn = probe_fn or _default_probe
    keyframe_fn = keyframe_fn or (lambda p, c: extract_keyframes(p, c))
    extract_audio_fn = extract_audio_fn or _default_extract_audio

    meta, warning = probe_fn(video_path, cfg)
    if warning:
        sink.emit(
            RunEvent(
                run_id=f"perception_{task.task_id}",
                event_id=f"perception_{task.task_id}:probe_warn",
                stage="perception",
                event_type="duration_out_of_window",
                level="warn",  # type: ignore[arg-type]
                task_id=task.task_id,
                payload={"warning": warning},
            )
        )

    def _warn(stage: str, exc: Exception) -> None:
        # One failing modality must not sink the whole ledger — partial credit is credit.
        # TEMP instrumentation: full type/message/traceback (behavior unchanged).
        sink.emit(RunEvent(
            run_id=f"perception_{task.task_id}",
            event_id=f"perception_{task.task_id}:{stage}_failed",
            stage="perception", event_type="stage_failed",
            level="warn",  # type: ignore[arg-type]
            task_id=task.task_id,
            payload={"stage": stage, "error": repr(exc)[:200],
                     "exc_type": type(exc).__name__, "exc_msg": str(exc),
                     "traceback": traceback.format_exc()},
        ))

    try:
        keyframes, motion_items = keyframe_fn(video_path, cfg)
    except Exception as exc:  # noqa: BLE001
        keyframes, motion_items = [], []
        _warn("keyframes", exc)
    try:
        audio_path = extract_audio_fn(video_path, cfg) if meta.has_audio else None
    except Exception as exc:  # noqa: BLE001
        audio_path = None
        _warn("audio_extract", exc)

    # OCR (PaddleOCR) has been removed; the vision model reads on-screen text directly from
    # the keyframes. No OCR items are produced, so evidence["ocr"] stays an empty list.
    ocr_items: list[EvidenceItem] = []
    per_frame: dict = {}
    try:
        speech_items, is_silent = transcribe(audio_path or "", meta, cfg,
                                             transcribe_fn=transcribe_fn)
    except Exception as exc:  # noqa: BLE001
        speech_items, is_silent = [], False
        _warn("speech", exc)
    try:
        audio_items = analyze_audio(audio_path or "", meta, cfg, feature_fn=audio_feature_fn)
    except Exception as exc:  # noqa: BLE001
        audio_items = []
        _warn("audio_events", exc)

    base_items: list[EvidenceItem] = (
        list(motion_items) + list(ocr_items) + list(speech_items) + list(audio_items)
    )
    return meta, list(keyframes), per_frame, base_items, is_silent, _warn


async def build_ledger(
    task: Task,
    cfg: Optional[PerceptionConfig] = None,
    *,
    vision_provider: Optional[VisionProvider] = None,
    video_path: Optional[str] = None,
    sink: Optional[EventSink] = None,
    probe_fn: Optional[Callable] = None,
    extract_audio_fn: Optional[Callable] = None,
    keyframe_fn: Optional[Callable] = None,
    transcribe_fn: Optional[Callable[[str, PerceptionConfig], list[SpeechSegment]]] = None,
    audio_feature_fn: Optional[Callable] = None,
) -> EvidenceLedger:
    """Turn a video file into an EvidenceLedger (public API, unchanged).

    When a ``vision_provider`` is supplied the per-frame VLM visual layer is appended, as
    before. CLARIS v2 leaves it None and reasons over the keyframes downstream instead —
    see ``build_perception``.
    """
    cfg = cfg or PerceptionConfig()
    sink = sink or NullSink()
    video_path = video_path or task.video_path

    meta, keyframes, per_frame, base_items, is_silent, _warn = await _collect_signals(
        task, cfg, sink, video_path=video_path, probe_fn=probe_fn,
        extract_audio_fn=extract_audio_fn, keyframe_fn=keyframe_fn,
        transcribe_fn=transcribe_fn, audio_feature_fn=audio_feature_fn,
    )

    if vision_provider is None:
        # No vision model reachable: ledger from speech + OCR + audio + motion only.
        visual_items: list[EvidenceItem] = []
        _warn("vision", RuntimeError("no vision model resolved; visual layer omitted"))
    else:
        try:
            visual_items = await describe_keyframes(
                keyframes, per_frame, vision_provider, cfg, sink=sink,
                run_id=f"perception_{task.task_id}",
            )
        except Exception as exc:  # noqa: BLE001
            visual_items = []
            _warn("vision", exc)

    all_items = base_items + list(visual_items)
    models = [it.source_model for it in all_items] + [cfg.vision_model, cfg.whisper_model]
    return assemble_ledger(task, meta, all_items, models, is_silent=is_silent)


async def build_perception(
    task: Task,
    cfg: Optional[PerceptionConfig] = None,
    *,
    video_path: Optional[str] = None,
    sink: Optional[EventSink] = None,
    probe_fn: Optional[Callable] = None,
    extract_audio_fn: Optional[Callable] = None,
    keyframe_fn: Optional[Callable] = None,
    transcribe_fn: Optional[Callable[[str, PerceptionConfig], list[SpeechSegment]]] = None,
    audio_feature_fn: Optional[Callable] = None,
) -> PerceptionBundle:
    """CLARIS v2 perception: a textual ledger (no per-frame VLM) plus the keyframes.

    The keyframes ride in the returned PerceptionBundle for the single downstream reasoning
    call; they never enter the frozen ledger.
    """
    cfg = cfg or PerceptionConfig()
    sink = sink or NullSink()
    video_path = video_path or task.video_path

    meta, keyframes, _per_frame, base_items, is_silent, _warn = await _collect_signals(
        task, cfg, sink, video_path=video_path, probe_fn=probe_fn,
        extract_audio_fn=extract_audio_fn, keyframe_fn=keyframe_fn,
        transcribe_fn=transcribe_fn, audio_feature_fn=audio_feature_fn,
    )
    models = [it.source_model for it in base_items] + [cfg.whisper_model]
    ledger = assemble_ledger(task, meta, base_items, models, is_silent=is_silent)
    return PerceptionBundle(ledger=ledger, keyframes=tuple(keyframes))


def _default_probe(video_path: str, cfg: PerceptionConfig):  # pragma: no cover
    from claris.core.perception.probe import probe  # noqa: PLC0415

    return probe(video_path, cfg)


def _default_extract_audio(video_path: str, cfg: PerceptionConfig) -> str:  # pragma: no cover
    import subprocess  # noqa: PLC0415
    import tempfile  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    # Write to a writable temp dir, never next to the input: /input may be mounted
    # read-only, so writing the extracted wav beside the video would fail.
    out = str(Path(tempfile.gettempdir()) / f"{Path(video_path).stem}.claris.wav")
    subprocess.run(
        ["ffmpeg", "-y", "-i", video_path, "-ac", "1", "-ar", "16000", out],
        capture_output=True,
        timeout=cfg.ffmpeg_timeout_s,
        check=True,
    )
    return out
