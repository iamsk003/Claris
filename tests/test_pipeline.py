"""End-to-end pipeline tests, fully offline via the deterministic mocks."""

from __future__ import annotations

import asyncio
import dataclasses

from claris.core.generation import GenerationConfig
from claris.core.pipeline import run_from_ledger
from claris.core.observability import ListSink
from claris.core.schema import ALL_STYLES, StyleName, Task
from eval.harness import load_golden_ledgers
from eval.mock_providers import mock_providers


def _task(ledger):
    return Task(task_id=ledger.task_id, video_path="x.mp4", styles=ALL_STYLES)


def test_gated_pipeline_produces_four_grounded_captions():
    ledger = load_golden_ledgers()[0]
    sink = ListSink()
    result = asyncio.run(
        run_from_ledger(ledger, _task(ledger), mock_providers(), sink=sink, use_gates=True)
    )
    assert set(result.captions) == set(ALL_STYLES)
    for cap in result.captions.values():
        assert cap.text and "dragon" not in cap.text.lower()  # gate_1 removed the ungrounded tier
        assert cap.score is not None                          # gate_2 scored the winner
    # The event stream carries every stage transition.
    types = {e.event_type for e in sink.events}
    assert {"generation_start", "gate_1_done", "gate_2_done", "gate_3_done", "pipeline_done"} <= types


def test_ungated_pipeline_takes_first_sample_ungrounded():
    ledger = load_golden_ledgers()[0]
    result = asyncio.run(
        run_from_ledger(ledger, _task(ledger), mock_providers(), use_gates=False)
    )
    assert set(result.captions) == set(ALL_STYLES)
    # First sample is the tier-0 caption, which carries the unsupported claim.
    assert any("dragon" in cap.text.lower() for cap in result.captions.values())
    assert all(cap.score is None for cap in result.captions.values())


def test_sequential_generation_passes_prior_captions():
    ledger = load_golden_ledgers()[0]
    task = Task(task_id=ledger.task_id, video_path="x.mp4", styles=ALL_STYLES)
    base = mock_providers()
    seen: list[tuple[str, str]] = []

    class RecordingGen:
        tier = base.gen_provider.tier
        model = base.gen_provider.model

        async def complete(self, **kw):
            seen.append(kw["prompt"])
            return await base.gen_provider.complete(**kw)

    providers = dataclasses.replace(base, gen_provider=RecordingGen())
    cfg = GenerationConfig(sequential=True, rate_per_sec=0.0)
    result = asyncio.run(run_from_ledger(ledger, task, providers, gen_config=cfg, use_gates=True))

    assert set(result.captions) == set(ALL_STYLES)
    assert all(c.score is not None for c in result.captions.values())
    # formal is generated first with no prior block; a later style carries one.
    assert "ALREADY WRITTEN" not in seen[0]
    assert any("ALREADY WRITTEN" in p for p in seen)


def test_submission_shape():
    ledger = load_golden_ledgers()[0]
    result = asyncio.run(run_from_ledger(ledger, _task(ledger), mock_providers(), use_gates=True))
    sub = result.to_submission()
    assert set(sub["captions"]) == {s.value for s in ALL_STYLES}
    assert all(isinstance(v, str) and v for v in sub["captions"].values())
