"""Fireworks AI provider — the default inference path.

OpenAI-compatible chat/completions over HTTPS. Every call is seeded,
temperature-explicit, JSON-mode by default, bounded by a timeout, and retried with
exponential backoff + jitter on transient failures (429 / 5xx / transport errors).

Serves Gemma (``ProviderTier.FIREWORKS_GEMMA``). If pointed at a non-Gemma model it
is the caller's responsibility to flag the degradation — this client just talks to
whatever ``model`` it is told to use.
"""

from __future__ import annotations

import base64
import os
import time
from typing import Optional

import httpx
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)

from claris.core.providers.base import ChatProvider, CompletionResult
from claris.core.schema import ProviderTier


def _message_content(data: dict) -> str:
    """Robustly pull the assistant text. Reasoning models (e.g. gpt-oss) may omit
    ``content`` when reasoning consumes the token budget, or split it into parts; never
    KeyError — return "" so the caller rejects cleanly instead of crashing the call."""
    msg = data["choices"][0]["message"]
    content = msg.get("content")
    if isinstance(content, str) and content.strip():
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict))
    rc = msg.get("reasoning_content")
    return rc if isinstance(rc, str) else ""


def _is_transient(exc: BaseException) -> bool:
    if isinstance(exc, httpx.TransportError):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code == 429 or exc.response.status_code >= 500
    return False


class FireworksProvider(ChatProvider):
    """Async Fireworks chat client."""

    tier = ProviderTier.FIREWORKS_GEMMA

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        client: Optional[httpx.AsyncClient] = None,
        max_attempts: int = 5,
    ) -> None:
        self.api_key = api_key or os.environ.get("FIREWORKS_API_KEY", "")
        self.base_url = (
            base_url
            or os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1")
        ).rstrip("/")
        self.model = model or os.environ.get(
            "CLARIS_GEMMA_GEN_MODEL", "accounts/fireworks/models/gemma-3"
        )
        self.max_attempts = max_attempts
        self._client = client
        self._owns_client = client is None

    @classmethod
    def from_env(cls) -> "FireworksProvider":
        return cls()

    def _get_client(self, timeout_s: float) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=timeout_s)
        return self._client

    async def aclose(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None

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
        if not self.api_key:
            raise RuntimeError(
                "FIREWORKS_API_KEY is not set; cannot call the Fireworks API."
            )
        use_model = model or self.model
        payload: dict = {
            "model": use_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "seed": seed,
            "max_tokens": max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        if "gemma" in use_model.lower():
            # Gemma 4 emits a thought preamble unless reasoning is disabled; that preamble
            # breaks strict-JSON parsing. Disable it on every Gemma call.
            payload["reasoning_effort"] = "none"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        client = self._get_client(timeout_s)
        url = f"{self.base_url}/chat/completions"

        @retry(
            retry=retry_if_exception(_is_transient),
            wait=wait_random_exponential(multiplier=0.5, max=20),
            stop=stop_after_attempt(self.max_attempts),
            reraise=True,
        )
        async def _do_request() -> httpx.Response:
            resp = await client.post(url, json=payload, headers=headers, timeout=timeout_s)
            resp.raise_for_status()
            return resp

        started = time.perf_counter()
        resp = await _do_request()
        latency_ms = (time.perf_counter() - started) * 1000.0

        data = resp.json()
        text = _message_content(data)
        usage = data.get("usage") or {}
        return CompletionResult(
            text=text,
            model=use_model,
            provider_tier=self.tier,
            seed=seed,
            temperature=temperature,
            latency_ms=latency_ms,
            tokens_in=usage.get("prompt_tokens"),
            tokens_out=usage.get("completion_tokens"),
        )


class FireworksVisionProvider(FireworksProvider):
    """Gemma 3 VLM over Fireworks: text plus base64-embedded image inputs."""

    def __init__(self, *, model: Optional[str] = None, **kwargs) -> None:
        super().__init__(
            model=model
            or os.environ.get(
                "CLARIS_GEMMA_VLM_MODEL", "accounts/fireworks/models/gemma-3-vlm"
            ),
            **kwargs,
        )

    async def complete(  # type: ignore[override]
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
    ) -> CompletionResult:
        if not self.api_key:
            raise RuntimeError("FIREWORKS_API_KEY is not set; cannot call the Fireworks API.")
        use_model = model or self.model
        content: list[dict] = [{"type": "text", "text": prompt}]
        for img in images:
            b64 = base64.b64encode(img).decode("ascii")
            content.append(
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
            )
        payload: dict = {
            "model": use_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": content},
            ],
            "temperature": temperature,
            "seed": seed,
            "max_tokens": max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        # Vision-capable models on this path (Gemma VLM, kimi, qwen) are reasoning models
        # whose chain-of-thought otherwise eats the token budget before the JSON is emitted,
        # producing empty/"no detail" descriptions. They all accept reasoning_effort=none.
        payload["reasoning_effort"] = "none"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        client = self._get_client(timeout_s)
        url = f"{self.base_url}/chat/completions"

        @retry(
            retry=retry_if_exception(_is_transient),
            wait=wait_random_exponential(multiplier=0.5, max=20),
            stop=stop_after_attempt(self.max_attempts),
            reraise=True,
        )
        async def _do_request() -> httpx.Response:
            resp = await client.post(url, json=payload, headers=headers, timeout=timeout_s)
            resp.raise_for_status()
            return resp

        started = time.perf_counter()
        resp = await _do_request()
        latency_ms = (time.perf_counter() - started) * 1000.0
        data = resp.json()
        usage = data.get("usage") or {}
        return CompletionResult(
            text=_message_content(data),
            model=use_model,
            provider_tier=self.tier,
            seed=seed,
            temperature=temperature,
            latency_ms=latency_ms,
            tokens_in=usage.get("prompt_tokens"),
            tokens_out=usage.get("completion_tokens"),
        )
