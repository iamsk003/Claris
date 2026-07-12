"""Batch captioning agent — portable across API keys.

Reads /input/tasks.json, writes /output/results.json incrementally, emits
/output/run_log.jsonl, exits 0. It resolves models by DISCOVERY (GET /v1/models + 1-token
capability probes), never by a hardcoded slug, so it runs against whatever the caller's key
exposes. The resolved model per role is written into results.json metadata and the run log.

A dedicated model deployment can be pinned with GEMMA_DEPLOYMENT_URL + GEMMA_DEPLOYMENT_MODEL.
``CLARIS_STRICT_MODELS=1`` makes a missing preferred model fatal instead of degrading.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Awaitable, Callable, Optional

import httpx

from claris.agent.config import AgentConfig
from claris.agent.discovery import ResolvedRoles, discover, is_gemma, resolve_roles
from claris.agent.fireworks_catalog import make_probes
from claris.core.observability import EventSink, JSONLSink
from claris.core.perception import PerceptionConfig
from claris.core.pipeline import Providers, run_pipeline
from claris.core.providers.base import CompletionResult, ProviderTier
from claris.core.schema import (
    RunEvent,
    StyledCaption,
    SubmissionFile,
    Task,
    TaskResult,
    utcnow,
)

PipelineFn = Callable[..., Awaitable[TaskResult]]

# Operational failures at the startup boundaries (env parsing, provider discovery) degrade
# to a clean exit 0. Programming errors (AttributeError, TypeError, KeyError, ...) are
# deliberately NOT caught so real bugs stay visible. An unwritable /output is intentionally
# left to fail: with no results.json the evaluator gets nothing, so it must not exit 0.
_STARTUP_ERRORS = (OSError, ValueError, httpx.HTTPError)


def _stderr(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


class _UnavailableProvider:
    """Stands in for an unresolved role. Any call fails cleanly -> the task degrades."""

    tier = ProviderTier.TEMPLATE

    def __init__(self, reason: str) -> None:
        self.model = "unavailable"
        self._reason = reason

    async def complete(self, **_kwargs) -> CompletionResult:
        raise RuntimeError(f"no model resolved for this role: {self._reason}")


async def resolve_providers(
    cfg: AgentConfig, *, client: httpx.AsyncClient,
    list_fn=None, probe_fn=None,
) -> tuple[Providers, ResolvedRoles]:
    """Build providers + role assignment from a deployment or from live discovery."""
    from claris.core.providers.fireworks import FireworksProvider, FireworksVisionProvider

    if cfg.has_deployment:
        gen = FireworksProvider(base_url=cfg.deployment_url, model=cfg.deployment_model,
                                api_key=cfg.api_key, client=client)
        vis = FireworksVisionProvider(base_url=cfg.deployment_url, model=cfg.deployment_model,
                                      api_key=cfg.api_key, client=client)
        m = cfg.deployment_model
        roles = ResolvedRoles(vlm=m, gen=m, gate1=m, critic=m,
                              gemma_path_used=is_gemma(m or ""), notes=("dedicated deployment",))
        return Providers(reasoning_provider=vis, gen_provider=gen, judge=gen, critic=gen,
                         vision_provider=vis), roles

    if not cfg.api_key:
        roles = ResolvedRoles(None, None, None, None, False,
                              ("no FIREWORKS_API_KEY; no models reachable",))
        un = _UnavailableProvider("no api key")
        return Providers(reasoning_provider=un, gen_provider=un, judge=un, critic=un,
                         vision_provider=None), roles

    lf, pf = make_probes(cfg.base_url, cfg.api_key, client)
    try:
        models = await discover(list_fn or lf, probe_fn or pf)
    except Exception as exc:  # noqa: BLE001 — discovery failure degrades, never crashes the agent
        roles = ResolvedRoles(
            None, None, None, None, False,
            (f"discovery failed ({repr(exc)[:160]}); no models reachable",),
        )
        un = _UnavailableProvider("discovery failed")
        return Providers(reasoning_provider=un, gen_provider=un, judge=un, critic=un,
                         vision_provider=None), roles
    roles = resolve_roles(models)

    def chat(model: Optional[str]):
        return (FireworksProvider(base_url=cfg.base_url, model=model, api_key=cfg.api_key,
                                  client=client) if model
                else _UnavailableProvider("no reachable chat model"))

    vision = (FireworksVisionProvider(base_url=cfg.base_url, model=roles.vlm,
                                      api_key=cfg.api_key, client=client) if roles.vlm else None)
    reasoning = vision if vision is not None else _UnavailableProvider("no multimodal model")
    return Providers(reasoning_provider=reasoning, gen_provider=chat(roles.gen),
                     judge=chat(roles.gate1), critic=chat(roles.critic),
                     vision_provider=vision), roles


def read_tasks(input_path: str) -> list[Task]:
    data = json.loads(Path(input_path).read_text())
    raw = data.get("tasks") if isinstance(data, dict) else data
    if not isinstance(raw, list):
        raise ValueError("tasks.json must be a list or an object with a 'tasks' list")
    return [Task.model_validate(t) for t in raw]


def _resolve_video(task: Task, input_dir: Path) -> Task:
    p = Path(task.video_path)
    return task if p.is_absolute() else task.model_copy(update={"video_path": str(input_dir / p)})


def _degraded_result(task: Task, run_id: str, error: str) -> TaskResult:
    caps = {
        style: StyledCaption(
            style=style,
            text="This clip could not be captioned reliably from the available evidence.",
            provider_tier=ProviderTier.TEMPLATE,
            degraded=True, degradation_reason=f"task_error: {error[:160]}",
        )
        for style in task.styles
    }
    return TaskResult(task_id=task.task_id, run_id=run_id, captions=caps,
                      degraded=True, error=error[:500])


def _write_results(output_path: str, run_id: str, results: list[TaskResult], metadata: dict) -> None:
    doc = SubmissionFile(run_id=run_id, results=[r.to_submission() for r in results],
                         metadata=metadata)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(doc.model_dump_json(indent=2))
    os.replace(tmp, out)


async def run_agent(
    providers: Providers,
    cfg: AgentConfig,
    *,
    metadata: dict,
    perception_config: Optional[PerceptionConfig] = None,
    use_gates: bool = True,
    pipeline_fn: Optional[PipelineFn] = None,
) -> int:
    """Run the task loop. Returns 0 (incl. degraded), 2 only on unreadable input."""
    pipeline_fn = pipeline_fn or run_pipeline
    run_id = f"agent_{utcnow().strftime('%Y%m%dT%H%M%S')}"

    try:
        tasks = read_tasks(cfg.input_path)
    except Exception as exc:  # noqa: BLE001 — unreadable input is the one hard failure
        _stderr(f"FATAL: cannot read tasks from {cfg.input_path}: {exc!r}")
        return 2

    log_path = Path(cfg.log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    sink: EventSink = JSONLSink(log_path)
    sink.emit(RunEvent(run_id=run_id, event_id=f"{run_id}:model_resolution", stage="agent",
                       event_type="model_resolution", payload=metadata))
    sink.emit(RunEvent(run_id=run_id, event_id=f"{run_id}:agent_start", stage="agent",
                       event_type="agent_start", payload={"n_tasks": len(tasks)}))

    input_dir = Path(cfg.input_path).parent
    results: list[TaskResult] = []
    loop = asyncio.get_running_loop()
    deadline = loop.time() + cfg.run_budget_s
    for task in tasks:
        sink.emit(RunEvent(run_id=run_id, event_id=f"{run_id}:{task.task_id}:task_start",
                           stage="agent", event_type="task_start", task_id=task.task_id))
        remaining = deadline - loop.time()
        if remaining <= 1.0:
            # Out of the run budget: degrade the rest immediately rather than overrun.
            sink.emit(RunEvent(run_id=run_id, event_id=f"{run_id}:{task.task_id}:budget",
                               stage="agent", event_type="budget_exhausted", level="warn",  # type: ignore[arg-type]
                               task_id=task.task_id))
            results.append(_degraded_result(task, run_id, "run budget exhausted"))
            _write_results(cfg.output_path, run_id, results, metadata)
            continue
        try:
            result = await asyncio.wait_for(
                pipeline_fn(_resolve_video(task, input_dir), providers,
                            perception_config=perception_config, sink=sink,
                            use_gates=use_gates, run_id=f"{run_id}_{task.task_id}"),
                timeout=min(cfg.task_timeout_s, remaining),
            )
        except Exception as exc:  # noqa: BLE001 — one bad clip must not sink the batch
            # TEMP instrumentation: surface the real failure (behavior unchanged).
            sink.emit(RunEvent(run_id=run_id, event_id=f"{run_id}:{task.task_id}:task_failed",
                               stage="agent", event_type="task_failed", level="error",  # type: ignore[arg-type]
                               task_id=task.task_id,
                               payload={"error": repr(exc),
                                        "exc_type": type(exc).__name__,
                                        "exc_msg": str(exc),
                                        "traceback": traceback.format_exc()}))
            result = _degraded_result(task, run_id, repr(exc))

        results.append(result)
        _write_results(cfg.output_path, run_id, results, metadata)  # incremental
        sink.emit(RunEvent(run_id=run_id, event_id=f"{run_id}:{task.task_id}:task_done",
                           stage="agent", event_type="task_done", task_id=task.task_id,
                           degraded=result.degraded))

    sink.emit(RunEvent(run_id=run_id, event_id=f"{run_id}:agent_done", stage="agent",
                       event_type="agent_done",
                       payload={"n_results": len(results),
                                "degraded": sum(1 for r in results if r.degraded)}))
    return 0


def _log_resolution(roles: ResolvedRoles) -> None:
    _stderr("[claris] model resolution:")
    _stderr(f"  reasoning  = {roles.vlm}")
    _stderr(f"  generation = {roles.gen}")
    _stderr(f"  gemma_path_used = {roles.gemma_path_used}")
    for note in roles.notes:
        _stderr(f"  note: {note}")


async def _amain() -> int:
    try:
        cfg = AgentConfig.from_env()
    except _STARTUP_ERRORS as exc:  # bad env config -> degrade, don't crash the container
        _stderr(f"[claris] startup: cannot build config, exiting 0: {exc!r}")
        return 0
    async with httpx.AsyncClient() as client:
        try:
            providers, roles = await resolve_providers(cfg, client=client)
        except _STARTUP_ERRORS as exc:  # provider/discovery init -> degrade, don't crash
            _stderr(f"[claris] startup: provider resolution failed, exiting 0: {exc!r}")
            return 0
        _log_resolution(roles)

        if cfg.strict:
            err = None if roles.gemma_path_used else "generation did not resolve to Gemma"
            if err is None:
                try:
                    await providers.gen_provider.complete(system="preflight", prompt="ok",
                                                          temperature=0.0, seed=1, max_tokens=1)
                except Exception as exc:  # noqa: BLE001
                    err = f"gen model unreachable: {exc!r}"
            if err:
                _stderr(f"FATAL (CLARIS_STRICT_MODELS=1): {err}")
                return 3

        _roles = roles.as_dict()
        metadata = {
            "resolved_roles": {"reasoning": _roles.get("vlm"),
                               "generation": _roles.get("gen"),
                               "gemma_path_used": _roles.get("gemma_path_used"),
                               "notes": _roles.get("notes")},
            "gemma_path_used": roles.gemma_path_used,
            "deployment": cfg.has_deployment,
            "strict": cfg.strict,
        }
        perception_config = PerceptionConfig(vision_model=roles.vlm or PerceptionConfig().vision_model)
        return await run_agent(providers, cfg, metadata=metadata,
                               perception_config=perception_config)


def main(argv: Optional[list[str]] = None) -> int:
    return asyncio.run(_amain())


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
