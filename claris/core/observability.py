"""Structured event logging — the append-only JSONL trail behind --replay.

Every LLM call and every rejection emits a ``RunEvent``. Sinks are pluggable so the
core stays I/O-light: the agent wires a ``JSONLSink`` to a run file, and tests use
``NullSink`` (or a ``ListSink``) to assert on emitted events without touching disk.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Protocol, runtime_checkable

from claris.core.schema import RunEvent

if TYPE_CHECKING:
    from claris.core.providers.base import CompletionResult


@runtime_checkable
class EventSink(Protocol):
    """Anything that can accept a structured event."""

    def emit(self, event: RunEvent) -> None: ...


class NullSink:
    """Drops events. The default when no logging is wired."""

    def emit(self, event: RunEvent) -> None:  # noqa: D401
        return None


class ListSink:
    """Collects events in memory. For tests and the API's live WebSocket feed."""

    def __init__(self) -> None:
        self.events: list[RunEvent] = []

    def emit(self, event: RunEvent) -> None:
        self.events.append(event)


class JSONLSink:
    """Appends events as JSON lines. Thread-safe for bounded concurrency."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def emit(self, event: RunEvent) -> None:
        line = event.to_jsonl()
        with self._lock:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")


def log_llm_call(
    sink: "EventSink",
    run_id: str,
    cost_stage: str,
    result: "CompletionResult",
    *,
    event_id: str,
    task_id: Optional[str] = None,
    extra: Optional[dict] = None,
) -> None:
    """Emit a billable ``llm_call`` event with tokens + model, tagged by cost stage.

    ``cost_stage`` is one of: perception_vision, generation, gate_1, gate_2, gate_3,
    held_out_judge. The spend meter groups on it, so every billable call must go through
    here (or emit the same shape) or it will be invisible on the bill.
    """
    sink.emit(
        RunEvent(
            run_id=run_id,
            event_id=event_id,
            stage=cost_stage,
            event_type="llm_call",
            task_id=task_id,
            model=result.model,
            provider_tier=result.provider_tier,
            seed=result.seed,
            temperature=result.temperature,
            latency_ms=result.latency_ms,
            tokens_in=result.tokens_in,
            tokens_out=result.tokens_out,
            payload={"cost_stage": cost_stage, **(extra or {})},
        )
    )


__all__ = ["EventSink", "NullSink", "ListSink", "JSONLSink", "log_llm_call"]
