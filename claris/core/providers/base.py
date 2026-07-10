"""Provider protocol + shared concurrency primitives.

Every inference backend implements the async ``ChatProvider`` interface. Generation
and verification depend only on this protocol, never on a concrete client, so a fake
provider can be injected in tests and the degradation ladder can swap tiers at runtime.

Contract obligations for every implementation:
  * seeded and temperature-explicit (both are required arguments),
  * bounded by a timeout,
  * returns a typed ``CompletionResult`` (never a raw dict).
Retry/backoff and structured logging live in the concrete clients and the callers.
"""

from __future__ import annotations

import asyncio
from typing import Optional, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from claris.core.schema import ProviderTier


class CompletionResult(BaseModel):
    """One completion plus the telemetry needed to log and replay the call."""

    text: str
    model: str
    provider_tier: ProviderTier
    seed: int
    temperature: float
    latency_ms: float = 0.0
    tokens_in: Optional[int] = None
    tokens_out: Optional[int] = None


@runtime_checkable
class ChatProvider(Protocol):
    """Async chat/completion interface shared by all backends."""

    tier: ProviderTier
    model: str

    async def complete(
        self,
        *,
        system: str,
        prompt: str,
        temperature: float,
        seed: int,
        model: Optional[str] = None,
        max_tokens: int = 512,
        timeout_s: float = 90.0,
        json_mode: bool = True,
    ) -> CompletionResult: ...


@runtime_checkable
class VisionProvider(Protocol):
    """Multimodal chat interface: same as ChatProvider plus image inputs.

    Images are passed as raw PNG/JPEG bytes; the concrete client is responsible for
    encoding them (e.g. base64 data URLs). Used by perception's Gemma 3 VLM stage.
    """

    tier: ProviderTier
    model: str

    async def complete(
        self,
        *,
        system: str,
        prompt: str,
        images: list[bytes],
        temperature: float,
        seed: int,
        model: Optional[str] = None,
        max_tokens: int = 1024,
        timeout_s: float = 90.0,
        json_mode: bool = True,
    ) -> CompletionResult: ...


class AsyncRateLimiter:
    """A global token-pacing limiter: at most ``rate_per_sec`` acquisitions/second.

    Enforces spacing between calls so 16 generations per clip do not stampede the
    Fireworks endpoint. ``rate_per_sec <= 0`` disables pacing.
    """

    def __init__(self, rate_per_sec: float) -> None:
        self._interval = 1.0 / rate_per_sec if rate_per_sec > 0 else 0.0
        self._lock = asyncio.Lock()
        self._next_at = 0.0

    async def acquire(self) -> None:
        if self._interval <= 0:
            return
        async with self._lock:
            loop = asyncio.get_running_loop()
            now = loop.time()
            wait = self._next_at - now
            if wait > 0:
                await asyncio.sleep(wait)
                now = loop.time()
            self._next_at = now + self._interval


__all__ = [
    "CompletionResult",
    "ChatProvider",
    "VisionProvider",
    "AsyncRateLimiter",
    "ProviderTier",
]
