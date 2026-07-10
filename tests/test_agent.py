"""Agent I/O-contract + portability tests. Mock providers only."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest

from claris.agent.config import AgentConfig
from claris.agent.main import resolve_providers, run_agent
from claris.core.schema import StyledCaption, Task, TaskResult
from eval.mock_providers import mock_providers

CLIPS = sorted(str(p) for p in Path("eval/clips").glob("*.mp4"))
FW = "accounts/fireworks/models/"


def _cfg(tmp_path: Path, tasks_name: str = "tasks.json", **kw) -> AgentConfig:
    return AgentConfig(
        input_path=str(tmp_path / tasks_name),
        output_path=str(tmp_path / "out" / "results.json"),
        log_path=str(tmp_path / "out" / "run_log.jsonl"),
        api_key="k", task_timeout_s=60.0, **kw,
    )


def _write_tasks(path: Path, video_paths: list[str]) -> None:
    path.write_text(json.dumps(
        {"tasks": [{"task_id": f"t{i}", "video_path": vp} for i, vp in enumerate(video_paths)]}
    ))


def _read(cfg: AgentConfig) -> dict:
    return json.loads(Path(cfg.output_path).read_text())


META = {"gemma_path_used": True, "resolved_roles": {"gen": "m"}}


# --------------------------------------------------------------- resolution


def test_resolve_providers_from_discovery():
    async def go():
        async with httpx.AsyncClient() as client:
            cfg = AgentConfig(input_path="i", output_path="o", log_path="l", api_key="k")

            async def list_fn():
                return [FW + "gemma-4-26b-a4b-it", FW + "gpt-oss-120b"]

            async def probe(mid):
                return (True, True, "gemma" in mid)  # gemma is vision-capable

            providers, roles = await resolve_providers(cfg, client=client,
                                                       list_fn=list_fn, probe_fn=probe)
            assert roles.gemma_path_used is True
            assert providers.vision_provider is not None
            assert roles.vlm.endswith("gemma-4-26b-a4b-it")

    asyncio.run(go())


def test_resolve_providers_no_key_is_unavailable_not_crash():
    async def go():
        async with httpx.AsyncClient() as client:
            cfg = AgentConfig(input_path="i", output_path="o", log_path="l", api_key=None)
            providers, roles = await resolve_providers(cfg, client=client)
            assert roles.gemma_path_used is False and roles.gen is None
            assert providers.vision_provider is None

    asyncio.run(go())


# --------------------------------------------------------------- I/O contract


@pytest.mark.skipif(len(CLIPS) != 4, reason="needs the 4 eval clips")
def test_smoke_over_four_clips_writes_metadata(tmp_path: Path):
    cfg = _cfg(tmp_path)
    _write_tasks(Path(cfg.input_path), CLIPS)
    rc = asyncio.run(run_agent(mock_providers(), cfg, metadata=META))
    assert rc == 0
    doc = _read(cfg)
    assert len(doc["results"]) == 4
    assert doc["metadata"]["gemma_path_used"] is True   # provenance in results.json
    for r in doc["results"]:
        assert set(r["captions"]) == {"formal", "sarcastic", "humorous_tech", "humorous_non_tech"}
    types = {json.loads(l)["event_type"] for l in Path(cfg.log_path).read_text().splitlines()}
    assert {"model_resolution", "agent_start", "agent_done"} <= types


def test_exit_zero_on_degraded_task(tmp_path: Path):
    cfg = _cfg(tmp_path)
    _write_tasks(Path(cfg.input_path), ["a.mp4", "b.mp4"])

    async def boom(task, providers, **kw):
        raise RuntimeError("no model reachable")

    rc = asyncio.run(run_agent(mock_providers(), cfg, metadata=META, pipeline_fn=boom))
    assert rc == 0
    doc = _read(cfg)
    assert len(doc["results"]) == 2
    for r in doc["results"]:
        assert len(r["captions"]) == 4  # degraded, never empty


def test_results_written_incrementally(tmp_path: Path):
    cfg = _cfg(tmp_path)
    _write_tasks(Path(cfg.input_path), ["a.mp4", "b.mp4", "c.mp4"])
    seen = []

    async def ok(task, providers, **kw):
        caps = {s: StyledCaption(style=s, text=f"{s.value} caption") for s in task.styles}
        p = Path(cfg.output_path)
        seen.append(len(json.loads(p.read_text())["results"]) if p.exists() else 0)
        return TaskResult(task_id=task.task_id, run_id="r", captions=caps)

    rc = asyncio.run(run_agent(mock_providers(), cfg, metadata=META, pipeline_fn=ok))
    assert rc == 0 and len(_read(cfg)["results"]) == 3
    assert seen == [0, 1, 2]


def test_unreadable_input_returns_nonzero(tmp_path: Path):
    cfg = _cfg(tmp_path, tasks_name="does_not_exist.json")
    rc = asyncio.run(run_agent(mock_providers(), cfg, metadata=META))
    assert rc == 2
    assert not Path(cfg.output_path).exists()


def test_strict_mode_fatal_without_gemma(monkeypatch, tmp_path: Path):
    from claris.agent import main as agent_main

    monkeypatch.delenv("FIREWORKS_API_KEY", raising=False)
    monkeypatch.setenv("CLARIS_STRICT_MODELS", "1")
    monkeypatch.setenv("CLARIS_INPUT", str(tmp_path / "tasks.json"))
    (tmp_path / "tasks.json").write_text('{"tasks": []}')
    # No key -> no Gemma resolves -> strict refuses to start (exit 3), before any task.
    assert agent_main.main([]) == 3
