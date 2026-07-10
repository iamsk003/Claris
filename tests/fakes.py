"""Offline test doubles for the generation subsystem — no network, deterministic."""

from __future__ import annotations

import json
from typing import Callable, Optional

from claris.core.providers.base import CompletionResult
from claris.core.schema import ProviderTier


class FakeProvider:
    """A ChatProvider that returns scripted JSON, keyed by a caller-supplied function.

    ``responder(system, prompt, temperature, seed)`` returns the raw model text. This
    lets a test emit valid captions, malformed JSON, or announced hallucinations at
    will, and assert on how generation reacts — all without a real model.
    """

    tier = ProviderTier.FIREWORKS_GEMMA

    def __init__(
        self,
        responder: Callable[[str, str, float, int], str],
        *,
        model: str = "fake-gemma",
    ) -> None:
        self._responder = responder
        self.model = model
        self.calls: list[dict] = []

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
    ) -> CompletionResult:
        self.calls.append({"temperature": temperature, "seed": seed})
        text = self._responder(system, prompt, temperature, seed)
        return CompletionResult(
            text=text,
            model=model or self.model,
            provider_tier=self.tier,
            seed=seed,
            temperature=temperature,
            latency_ms=1.0,
            tokens_in=10,
            tokens_out=20,
        )


def valid_json(caption: str, cited: list[str]) -> str:
    return json.dumps({"caption": caption, "cited_evidence_ids": cited})


class FakeVisionProvider:
    """A VisionProvider returning scripted text. Records image count per call.

    ``responder(prompt, n_images, attempt)`` returns the raw model text; ``attempt``
    starts at 0 and increments on the repair retry, so a test can fail the first parse
    and succeed on the second.
    """

    tier = ProviderTier.FIREWORKS_GEMMA

    def __init__(self, responder, *, model: str = "fake-gemma-vlm") -> None:
        self._responder = responder
        self.model = model
        self.calls: list[int] = []

    async def complete(
        self,
        *,
        system: str,
        prompt: str,
        images,
        temperature: float,
        seed: int,
        model=None,
        max_tokens: int = 1024,
        timeout_s: float = 90.0,
        json_mode: bool = True,
    ) -> CompletionResult:
        attempt = len(self.calls)
        self.calls.append(len(images))
        text = self._responder(prompt, len(images), attempt)
        return CompletionResult(
            text=text,
            model=model or self.model,
            provider_tier=self.tier,
            seed=seed,
            temperature=temperature,
            latency_ms=1.0,
        )
