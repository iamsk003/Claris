"""ASR via faster-whisper, in-container, always.

There is deliberately no "Fireworks Whisper if available" branch: ASR runs locally so the
perception stack needs no external inference service. Weights are preloaded at Docker build
time.

The ledger must distinguish three states, so we never emit a bare empty list:
  * has speech       -> one EvidenceItem per utterance,
  * silent with audio -> a single explicit "no speech detected" item (is_silent),
  * no audio track    -> a single explicit "clip has no audio track" item.

``segments_to_items`` and ``build_speech_items`` are pure and unit tested; the
faster-whisper call is isolated behind an injectable ``transcribe_fn``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from claris.core.perception.config import PerceptionConfig
from claris.core.schema import EvidenceItem, VideoMeta

SOURCE_MODEL = "faster-whisper"

# Whisper hallucinates on non-speech audio (ASMR, music, silence), emitting near-zero
# confidence garbage like "~!" or "♪". Drop segments that are too low-confidence or carry
# no actual word characters, so a non-speech clip resolves to "no speech" not to noise.
MIN_SPEECH_CONFIDENCE = 0.3
_HAS_WORD = re.compile(r"[A-Za-z0-9]")


def _is_real_speech(text: str, confidence: float, min_confidence: float) -> bool:
    return confidence >= min_confidence and bool(_HAS_WORD.search(text or ""))


@dataclass(frozen=True)
class SpeechSegment:
    t_start: float
    t_end: float
    text: str
    confidence: float


def segments_to_items(
    segments: list[SpeechSegment], source_model: str = SOURCE_MODEL
) -> list[EvidenceItem]:
    """Turn transcript segments into speech EvidenceItems. Pure."""
    items: list[EvidenceItem] = []
    for i, seg in enumerate(segments, start=1):
        text = seg.text.strip()
        if not text:
            continue
        items.append(
            EvidenceItem(
                id=f"speech_{i:03d}",
                kind="speech",
                t_start=seg.t_start,
                t_end=max(seg.t_end, seg.t_start),
                content=text,
                confidence=seg.confidence,
                source_model=source_model,
            )
        )
    return items


def build_speech_items(
    segments: list[SpeechSegment],
    meta: VideoMeta,
    source_model: str = SOURCE_MODEL,
    min_confidence: float = MIN_SPEECH_CONFIDENCE,
) -> tuple[list[EvidenceItem], bool]:
    """Return (items, is_silent). Never returns an empty list ambiguously.

    Segments below ``min_confidence`` or with no word characters are dropped as Whisper
    hallucinations; if none survive, the clip is reported silent rather than as noise.
    ``is_silent`` is True only when there is an audio track but no real speech in it.
    """
    if not meta.has_audio:
        item = EvidenceItem(
            id="speech_000",
            kind="speech",
            t_start=0.0,
            t_end=meta.duration_s,
            content="Clip has no audio track.",
            confidence=1.0,
            source_model=source_model,
        )
        return [item], False

    real = [s for s in segments if _is_real_speech(s.text, s.confidence, min_confidence)]
    items = segments_to_items(real, source_model)
    if items:
        return items, False

    silent = EvidenceItem(
        id="speech_000",
        kind="speech",
        t_start=0.0,
        t_end=meta.duration_s,
        content="No speech detected in this clip.",
        confidence=0.9,
        source_model=source_model,
    )
    return [silent], True


def _faster_whisper_transcribe(
    audio_path: str, cfg: PerceptionConfig
) -> list[SpeechSegment]:  # pragma: no cover - needs the model + audio
    from faster_whisper import WhisperModel  # noqa: PLC0415

    model = WhisperModel(cfg.whisper_model, device="auto", compute_type="int8")
    # vad_filter drops non-speech audio before decoding; condition_on_previous_text=False
    # stops the model looping hallucinated text. Both suppress the "~!"/"♪" garbage that
    # Whisper otherwise invents on ASMR/music/silence.
    segments, _info = model.transcribe(
        audio_path, word_timestamps=True, vad_filter=True,
        condition_on_previous_text=False,
    )
    out: list[SpeechSegment] = []
    for seg in segments:
        # avg_logprob is roughly [-1, 0]; map to a [0, 1] confidence.
        conf = max(0.0, min(1.0, 1.0 + getattr(seg, "avg_logprob", -0.3)))
        out.append(
            SpeechSegment(
                t_start=float(seg.start),
                t_end=float(seg.end),
                text=seg.text,
                confidence=round(conf, 3),
            )
        )
    return out


def transcribe(
    audio_path: str,
    meta: VideoMeta,
    cfg: Optional[PerceptionConfig] = None,
    *,
    transcribe_fn=None,
) -> tuple[list[EvidenceItem], bool]:
    """Transcribe audio into speech EvidenceItems. Returns (items, is_silent).

    ``transcribe_fn(audio_path, cfg) -> list[SpeechSegment]`` is injectable so unit
    tests exercise the silent/no-audio logic without the model or the network.
    """
    cfg = cfg or PerceptionConfig()
    if not meta.has_audio:
        return build_speech_items([], meta)
    fn = transcribe_fn or (lambda p, c: _faster_whisper_transcribe(p, c))
    segments = fn(audio_path, cfg)
    return build_speech_items(segments, meta)
