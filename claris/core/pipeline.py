"""The CLARIS v2 core pipeline. One engine, one event stream, one reasoning call.

    probe -> shots -> (speech || ocr || audio_events || motion) -> PerceptionBundle
          -> ONE multimodal reasoning call -> four StyledCaptions -> TaskResult

Perception is delegated to ``claris.core.perception.build_perception`` (textual ledger +
keyframes). Generation, grounding, and the four styles are produced by a single multimodal
call in ``claris.core.reasoning`` — there is no rejection sampling, critic, or tone-separation
pass. Providers are injected, so the whole pipeline runs offline in tests with fakes and makes
zero network calls.

The public seam is unchanged: ``run_from_ledger(ledger, task, providers, ...)`` and
``run_pipeline(task, providers, ...)`` keep their signatures. ``run_from_ledger`` gains an
optional ``keyframes`` argument; callers that pass none still get a (text-only) result.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from claris.core.observability import EventSink, NullSink
from claris.core.perception import PerceptionConfig, build_perception
from claris.core.perception.bundle import PerceptionBundle
from claris.core.perception.shots import Keyframe
from claris.core.providers.base import ChatProvider, VisionProvider
from claris.core.reasoning import reason_over_clip
from claris.core.schema import EvidenceLedger, RunEvent, Task, TaskResult, utcnow


@dataclass
class Providers:
    """The inference backends the pipeline needs.

    v2 requires only ``reasoning_provider`` — a generic multimodal chat model (Kimi, Qwen,
    GLM, MiniMax, Gemma, …). The remaining fields are kept optional so existing construction
    sites (``resolve_providers``) stay valid; they are no longer on the runtime path.
    """

    reasoning_provider: Optional[VisionProvider] = None
    gen_provider: Optional[ChatProvider] = None
    judge: Optional[ChatProvider] = None
    critic: Optional[ChatProvider] = None
    vision_provider: Optional[VisionProvider] = None
    embed_fn: Optional[object] = None


def _emit(sink: EventSink, run_id: str, task_id: str, event_type: str, **payload) -> None:
    sink.emit(RunEvent(run_id=run_id, event_id=f"{run_id}:{event_type}", stage="pipeline",
                       event_type=event_type, task_id=task_id, payload=payload))


async def run_from_ledger(
    ledger: EvidenceLedger,
    task: Task,
    providers: Providers,
    *,
    keyframes: Optional[tuple[Keyframe, ...]] = None,
    perception_config: Optional[PerceptionConfig] = None,
    sink: Optional[EventSink] = None,
    run_id: Optional[str] = None,
    **_ignored,
) -> TaskResult:
    """Reason over a ledger (+ optional keyframes) into a TaskResult.

    ``**_ignored`` absorbs retired v1 knobs (gen_config, ver_config, registry, use_gates)
    so existing callers keep working without change.
    """
    sink = sink or NullSink()
    run_id = run_id or f"pipe_{task.task_id}_{utcnow().strftime('%Y%m%dT%H%M%S')}"
    bundle = PerceptionBundle(ledger=ledger, keyframes=tuple(keyframes or ()))
    provider = providers.reasoning_provider or providers.vision_provider
    return await reason_over_clip(
        bundle, task, provider, cfg=perception_config, sink=sink, run_id=run_id,
    )


async def run_pipeline(
    task: Task,
    providers: Providers,
    *,
    perception_config: Optional[PerceptionConfig] = None,
    sink: Optional[EventSink] = None,
    run_id: Optional[str] = None,
    **perception_kwargs,
) -> TaskResult:
    """Full pipeline from a video file: perception (bundle) then the single reasoning call."""
    sink = sink or NullSink()
    run_id = run_id or f"pipe_{task.task_id}_{utcnow().strftime('%Y%m%dT%H%M%S')}"
    for _retired in ("use_gates", "gen_config", "ver_config", "registry"):
        perception_kwargs.pop(_retired, None)
    _emit(sink, run_id, task.task_id, "perception_start")
    bundle = await build_perception(task, perception_config, sink=sink, **perception_kwargs)
    _emit(sink, run_id, task.task_id, "perception_done",
          items=len(bundle.ledger.items), coverage=bundle.ledger.coverage,
          keyframes=len(bundle.keyframes))

    return await run_from_ledger(
        bundle.ledger, task, providers, keyframes=bundle.keyframes,
        perception_config=perception_config, sink=sink, run_id=run_id,
    )


__all__ = ["Providers", "run_from_ledger", "run_pipeline"]
