"""Local vLLM provider (STUB — do not implement yet).

The most-preferred tier (``ProviderTier.LOCAL_GEMMA``) when a local GPU running vLLM is
available. Serves a LoRA-adapted Gemma. Absent a local GPU, the ladder falls through to
Fireworks.
"""

from __future__ import annotations

from claris.core.schema import ProviderTier


class LocalVLLMProvider:  # pragma: no cover
    """Local Gemma via vLLM. Not implemented."""

    tier = ProviderTier.LOCAL_GEMMA

    def complete(self, *, prompt: str, model: str, temperature: float, seed: int) -> str:
        raise NotImplementedError("providers.local_vllm.LocalVLLMProvider is a stub")
