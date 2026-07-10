"""Agent configuration.

The shipped container resolves models by discovery (see ``discovery``), so no model ID is
required to start. Two ways to force the Gemma path:
  * a dedicated deployment (``GEMMA_DEPLOYMENT_URL`` + ``GEMMA_DEPLOYMENT_MODEL``) — how
    eval/ reaches our Gemma while the container stays portable;
  * ``CLARIS_STRICT_MODELS=1`` — restores fatal behaviour so a misconfigured local run that
    expected Gemma fails loudly instead of silently degrading. Default off.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping, Optional


class AgentConfigError(RuntimeError):
    """Raised for a fatal misconfiguration (only under strict mode)."""


@dataclass(frozen=True)
class AgentConfig:
    input_path: str
    output_path: str
    log_path: str
    api_key: Optional[str] = None
    base_url: str = "https://api.fireworks.ai/inference/v1"
    deployment_url: Optional[str] = None
    deployment_model: Optional[str] = None
    strict: bool = False
    # Bound the whole run and each task so the container always exits inside a ~10 minute
    # window rather than running unbounded.
    run_budget_s: float = 540.0
    task_timeout_s: float = 540.0

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "AgentConfig":
        env = env if env is not None else os.environ
        return cls(
            input_path=env.get("CLARIS_INPUT", "/input/tasks.json"),
            output_path=env.get("CLARIS_OUTPUT", "/output/results.json"),
            log_path=env.get("CLARIS_RUN_LOG", "/output/run_log.jsonl"),
            api_key=env.get("FIREWORKS_API_KEY"),
            base_url=env.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1"),
            deployment_url=env.get("GEMMA_DEPLOYMENT_URL"),
            deployment_model=env.get("GEMMA_DEPLOYMENT_MODEL"),
            strict=env.get("CLARIS_STRICT_MODELS") == "1",
            run_budget_s=float(env.get("CLARIS_RUN_BUDGET_S", "540")),
            task_timeout_s=float(env.get("CLARIS_TASK_TIMEOUT_S", "540")),
        )

    @property
    def has_deployment(self) -> bool:
        return bool(self.deployment_url and self.deployment_model)
