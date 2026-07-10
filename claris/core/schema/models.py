"""CLARIS schema — the single source of truth.

Every boundary between modules is a Pydantic v2 model defined here. No raw dicts
cross module lines. If two subsystems need to agree on a shape, that shape lives
in this file and nowhere else.

Design rules obeyed by this module:
  * EvidenceItem and EvidenceLedger are frozen (immutable) once constructed. The
    ledger is the ground truth the whole pipeline is checked against; it must not
    be mutable after perception emits it.
  * IDs are content-stable and human-readable (``ev_0001``, ``cand_...``).
  * Every score is bounded to [0, 1]; every timestamp is seconds (float).
  * Anything that gets logged or cached derives its key from these models via the
    hashing helpers at the bottom of this file.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

# --------------------------------------------------------------------------- #
# Time / hashing helpers
# --------------------------------------------------------------------------- #


def utcnow() -> datetime:
    """Timezone-aware UTC now. Never use naive datetimes in this codebase."""
    return datetime.now(timezone.utc)


def sha256_str(text: str) -> str:
    """Hex sha256 of a UTF-8 string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def params_hash(params: dict[str, Any]) -> str:
    """Stable hash of a parameter dict, for the content-addressed cache.

    Keys are sorted and the encoding is canonical, so the same params always
    produce the same key regardless of insertion order.
    """
    canonical = json.dumps(params, sort_keys=True, separators=(",", ":"), default=str)
    return sha256_str(canonical)


def estimate_tokens(text: str) -> int:
    """Rough token count (~4 chars/token) for budgeting prompts without a tokenizer dep."""
    return (len(text) + 3) // 4


# --------------------------------------------------------------------------- #
# Enums — closed vocabularies
# --------------------------------------------------------------------------- #


class StyleName(str, Enum):
    """The four caption styles. This ordering is canonical."""

    FORMAL = "formal"
    SARCASTIC = "sarcastic"
    HUMOROUS_TECH = "humorous_tech"
    HUMOROUS_NON_TECH = "humorous_non_tech"


ALL_STYLES: tuple[StyleName, ...] = (
    StyleName.FORMAL,
    StyleName.SARCASTIC,
    StyleName.HUMOROUS_TECH,
    StyleName.HUMOROUS_NON_TECH,
)


class EvidenceKind(str, Enum):
    """What a piece of evidence is derived from."""

    SPEECH = "speech"          # ASR transcript segment
    VISUAL = "visual"          # VLM keyframe description
    OCR = "ocr"                # on-screen text
    AUDIO_EVENT = "audio_event"  # non-speech audio (music, laughter, crash)
    MOTION = "motion"          # detected motion / action / scene change


class ProviderTier(str, Enum):
    """The graceful-degradation ladder. Lower index = more preferred.

    Anything below LOCAL_GEMMA / FIREWORKS_GEMMA means we are off the Gemma path
    and MUST set ``degraded=True`` on the output, with a logged reason.
    """

    LOCAL_GEMMA = "local_gemma"                # local vLLM GPU deployment
    FIREWORKS_GEMMA = "fireworks_gemma"        # default hosted path
    FIREWORKS_NON_GEMMA = "fireworks_non_gemma"  # explicit, logged degradation
    TEMPLATE = "template"                      # deterministic last resort


class EventLevel(str, Enum):
    INFO = "info"
    WARN = "warn"
    ERROR = "error"


# --------------------------------------------------------------------------- #
# Perception layer
# --------------------------------------------------------------------------- #


class VideoMeta(BaseModel):
    """Immutable facts about the input video file itself."""

    model_config = ConfigDict(frozen=True)

    video_sha256: str = Field(..., description="Content hash; primary cache key.")
    duration_s: float = Field(..., ge=0.0)
    fps: Optional[float] = Field(default=None, ge=0.0)
    width: Optional[int] = Field(default=None, ge=0)
    height: Optional[int] = Field(default=None, ge=0)
    has_audio: bool = True
    container: Optional[str] = None
    video_codec: Optional[str] = None
    audio_codec: Optional[str] = None


class EvidenceItem(BaseModel):
    """One immutable, timestamped, ID-addressed fact about the clip.

    This is the atom of the Evidence Ledger. Captions are checked claim-by-claim
    against these; a claim with no supporting EvidenceItem.id is a hallucination.
    """

    model_config = ConfigDict(frozen=True)

    id: str = Field(..., description="Stable ID, e.g. 'ev_0001'.")
    kind: EvidenceKind
    t_start: float = Field(..., ge=0.0, description="Seconds from clip start.")
    t_end: float = Field(..., ge=0.0, description="Seconds from clip start.")
    content: str = Field(..., min_length=1)
    confidence: float = Field(..., ge=0.0, le=1.0)
    source_model: str = Field(..., description="Model/tool that produced this.")

    @field_validator("id")
    @classmethod
    def _id_nonempty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("EvidenceItem.id must be non-empty")
        return v

    @model_validator(mode="after")
    def _time_ordered(self) -> "EvidenceItem":
        if self.t_end < self.t_start:
            raise ValueError(
                f"t_end ({self.t_end}) must be >= t_start ({self.t_start}) "
                f"for evidence {self.id}"
            )
        return self


class ModalityFlags(BaseModel):
    """Which signal channels the perception stack actually recovered for a clip.

    ``is_silent`` distinguishes "we transcribed and there was no speech" from a
    missing speech modality, which downstream prompts must not confuse.
    """

    model_config = ConfigDict(frozen=True)

    has_speech: bool = False
    has_ocr: bool = False
    has_audio_event: bool = False
    has_visual: bool = False
    has_motion: bool = False
    is_silent: bool = False

    def active(self) -> tuple[str, ...]:
        names = (
            ("speech", self.has_speech),
            ("ocr", self.has_ocr),
            ("audio_event", self.has_audio_event),
            ("visual", self.has_visual),
            ("motion", self.has_motion),
        )
        return tuple(name for name, on in names if on)


class EvidenceLedger(BaseModel):
    """The immutable ground truth for one clip.

    Emitted once by perception, then read-only for the rest of the pipeline.
    Frozen at the model level; construct it fully, then hand it around.
    """

    model_config = ConfigDict(frozen=True)

    ledger_id: str
    task_id: str
    video_sha256: str
    video_meta: VideoMeta
    items: tuple[EvidenceItem, ...] = Field(default_factory=tuple)
    perception_models: tuple[str, ...] = Field(default_factory=tuple)
    coverage: float = Field(default=0.0, ge=0.0, le=1.0)
    modality_flags: ModalityFlags = Field(default_factory=ModalityFlags)
    created_at: datetime = Field(default_factory=utcnow)

    @model_validator(mode="after")
    def _unique_ids(self) -> "EvidenceLedger":
        ids = [it.id for it in self.items]
        if len(ids) != len(set(ids)):
            dupes = {i for i in ids if ids.count(i) > 1}
            raise ValueError(f"Duplicate evidence ids in ledger: {sorted(dupes)}")
        return self

    def get(self, evidence_id: str) -> Optional[EvidenceItem]:
        """Return the EvidenceItem with this id, or None."""
        for it in self.items:
            if it.id == evidence_id:
                return it
        return None

    def by_kind(self, kind: EvidenceKind) -> tuple[EvidenceItem, ...]:
        return tuple(it for it in self.items if it.kind == kind)

    def to_prompt_block(self) -> str:
        """Render the ledger as the grounding block for a generation prompt.

        Deterministic (items sorted by start time then id) so the prompt hash is
        stable for caching and --replay. Every line is ID-addressed so the model
        can cite ``ev_xxxx`` and the hallucination gate can check the citation.
        """
        modalities = ", ".join(self.modality_flags.active()) or "none"
        header = (
            f"VIDEO EVIDENCE LEDGER "
            f"(duration {self.video_meta.duration_s:.0f}s, {len(self.items)} items, "
            f"coverage {self.coverage:.2f})"
        )
        lines = [header, f"modalities: {modalities}"]
        if self.modality_flags.is_silent:
            lines.append("note: clip is silent (no speech present, not a failure).")
        for it in sorted(self.items, key=lambda x: (x.t_start, x.id)):
            lines.append(
                f"[{it.id}] {it.kind.value} {it.t_start:.1f}-{it.t_end:.1f}s "
                f"({it.confidence:.2f}): {it.content}"
            )
        return "\n".join(lines)

    @property
    def evidence_ids(self) -> frozenset[str]:
        return frozenset(it.id for it in self.items)

    def has(self, evidence_id: str) -> bool:
        return evidence_id in self.evidence_ids


# --------------------------------------------------------------------------- #
# Generation layer
# --------------------------------------------------------------------------- #


class CaptionCandidate(BaseModel):
    """One sampled caption for one style, before it has been scored/selected.

    Rejection sampling produces N of these per style; the critic scores each and
    the argmax becomes a StyledCaption.
    """

    candidate_id: str
    style: StyleName
    text: str = Field(..., min_length=1)
    evidence_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Evidence IDs this caption's claims are grounded in.",
    )
    temperature: float = Field(..., ge=0.0)
    seed: int
    model: str
    provider_tier: ProviderTier = ProviderTier.FIREWORKS_GEMMA
    created_at: datetime = Field(default_factory=utcnow)


# --------------------------------------------------------------------------- #
# Verification layer
# --------------------------------------------------------------------------- #


class CritiqueScore(BaseModel):
    """The internal critic's judgement of a candidate, scored 1-5 per dimension.

    Four dimensions: accuracy, tone_fidelity, style_distinctness, naturalness. The
    weighting lives in VerificationConfig, not here — the critic computes ``overall``
    with the config weights and stores it, so the objective can be retuned without
    touching the schema. Per-dimension reasoning strings drive the frontend Judge View.
    """

    accuracy: float = Field(..., ge=1.0, le=5.0)
    tone_fidelity: float = Field(..., ge=1.0, le=5.0)
    style_distinctness: float = Field(..., ge=1.0, le=5.0)
    naturalness: float = Field(..., ge=1.0, le=5.0)
    overall: float = Field(..., ge=1.0, le=5.0, description="Config-weighted aggregate.")
    accuracy_reason: str = ""
    tone_reason: str = ""
    distinctness_reason: str = ""
    naturalness_reason: str = ""
    unsupported_claims: tuple[str, ...] = Field(default_factory=tuple)
    critic_model: str = "gemma-critic"

    @property
    def has_hallucination(self) -> bool:
        return len(self.unsupported_claims) > 0


class StyledCaption(BaseModel):
    """The selected, verified caption for one style. The final emitted output."""

    style: StyleName
    text: str = Field(..., min_length=1)
    candidate_id: Optional[str] = Field(
        default=None, description="ID of the winning candidate; None for template."
    )
    evidence_ids: tuple[str, ...] = Field(default_factory=tuple)
    score: Optional[CritiqueScore] = None
    provider_tier: ProviderTier = ProviderTier.FIREWORKS_GEMMA
    degraded: bool = False
    degradation_reason: Optional[str] = None
    degraded_ungrounded: bool = Field(
        default=False,
        description="True when emitted as best-of-batch after the grounding gate wiped "
        "the style; the caption is unverified, never absent.",
    )

    @model_validator(mode="after")
    def _degradation_has_reason(self) -> "StyledCaption":
        if self.degraded and not self.degradation_reason:
            raise ValueError("degraded=True requires a degradation_reason")
        return self


# --------------------------------------------------------------------------- #
# Task boundary — what goes in, what comes out
# --------------------------------------------------------------------------- #


class Task(BaseModel):
    """One unit of work read from /input/tasks.json."""

    task_id: str
    video_path: str = Field(..., description="Path to the clip, relative to /input.")
    video_sha256: Optional[str] = Field(
        default=None, description="Filled in after hashing the file."
    )
    styles: tuple[StyleName, ...] = Field(default=ALL_STYLES)
    notes: Optional[str] = None


class TaskResult(BaseModel):
    """The complete result for one Task. Serialized into /output/results.json."""

    task_id: str
    run_id: str
    video_sha256: Optional[str] = None
    ledger_id: Optional[str] = None
    captions: dict[StyleName, StyledCaption] = Field(default_factory=dict)
    degraded: bool = False
    error: Optional[str] = None
    created_at: datetime = Field(default_factory=utcnow)

    def to_submission(self) -> dict[str, Any]:
        """The exact output shape emitted per task.

        Kept intentionally small and stable: task_id plus one string per style.
        Everything else (scores, evidence, tiers) stays in the logs, not here.
        """
        return {
            "task_id": self.task_id,
            "captions": {
                style.value: cap.text for style, cap in self.captions.items()
            },
        }


class SubmissionFile(BaseModel):
    """The top-level /output/results.json document."""

    run_id: str
    results: list[dict[str, Any]]
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Resolved model per role, gemma_path_used, etc. — provenance for judges.",
    )
    created_at: datetime = Field(default_factory=utcnow)


# --------------------------------------------------------------------------- #
# Observability — the structured event log (JSONL)
# --------------------------------------------------------------------------- #


class RunEvent(BaseModel):
    """One structured log line. Append-only JSONL; the basis for --replay.

    Every LLM call emits at least one of these with the fields needed to
    reconstruct the call deterministically: prompt_hash, model, seed, temperature.
    """

    run_id: str
    event_id: str
    ts: datetime = Field(default_factory=utcnow)
    stage: str = Field(..., description="perception|generation|verification|agent|...")
    event_type: str = Field(..., description="e.g. 'llm_call', 'cache_hit', 'error'.")
    level: EventLevel = EventLevel.INFO
    task_id: Optional[str] = None

    # LLM-call telemetry
    model: Optional[str] = None
    provider_tier: Optional[ProviderTier] = None
    prompt_hash: Optional[str] = None
    seed: Optional[int] = None
    temperature: Optional[float] = None
    latency_ms: Optional[float] = None
    tokens_in: Optional[int] = None
    tokens_out: Optional[int] = None

    # Degradation / free-form
    degraded: bool = False
    payload: dict[str, Any] = Field(default_factory=dict)

    def to_jsonl(self) -> str:
        return self.model_dump_json()


__all__ = [
    # helpers
    "utcnow",
    "sha256_str",
    "params_hash",
    "estimate_tokens",
    # enums
    "StyleName",
    "ALL_STYLES",
    "EvidenceKind",
    "ProviderTier",
    "EventLevel",
    # perception
    "VideoMeta",
    "EvidenceItem",
    "EvidenceLedger",
    "ModalityFlags",
    # generation
    "CaptionCandidate",
    # verification
    "CritiqueScore",
    "StyledCaption",
    # task boundary
    "Task",
    "TaskResult",
    "SubmissionFile",
    # observability
    "RunEvent",
]
