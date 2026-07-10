"""Cost meter tests — pricing and aggregation are deterministic and offline."""

from __future__ import annotations

from eval.spend import deployment_session_event, load_prices, price_per_1m, report

PRICES = load_prices()


def _ev(stage, model, tin, tout):
    return {"event_type": "llm_call", "model": model,
            "tokens_in": tin, "tokens_out": tout, "payload": {"cost_stage": stage}}


def test_price_lookup_known_and_default():
    assert price_per_1m("accounts/fireworks/models/gemma-3-4b", PRICES) == 0.20
    assert price_per_1m("accounts/fireworks/models/gemma-3", PRICES) == 0.90
    # Unknown model falls back to the conservative default, never free.
    assert price_per_1m("accounts/fireworks/models/who-knows", PRICES) == PRICES["default_price_per_1m"]


def test_report_groups_and_costs():
    events = [
        _ev("generation", "accounts/fireworks/models/gemma-3", 1_000_000, 0),   # $0.90
        _ev("gate_1", "accounts/fireworks/models/gemma-3-4b", 500_000, 500_000),  # 1M @0.20 = $0.20
        _ev("held_out_judge", "accounts/fireworks/models/llama-v3p1-70b-instruct", 0, 100_000),  # $0.09
        {"event_type": "candidate_scored", "model": "x", "payload": {}},  # ignored (not llm_call)
    ]
    rep = report(events, PRICES)
    assert rep["by_stage"]["generation"]["usd"] == 0.9
    assert rep["by_stage"]["gate_1"]["usd"] == 0.2
    assert round(rep["by_stage"]["held_out_judge"]["usd"], 4) == 0.09
    assert rep["total_tokens"] == 2_100_000
    assert round(rep["total_usd"], 4) == 1.19
    assert rep["by_stage"]["generation"]["calls"] == 1


def test_empty_log_is_zero():
    rep = report([], PRICES)
    assert rep["total_usd"] == 0.0 and rep["total_tokens"] == 0
    assert rep["deployment"]["usd"] == 0.0


def test_deployment_gpu_hour_pricing():
    # 2 hours of an H100 deployment at $7/hr = $14, independent of tokens (idle bills too).
    rep = report([deployment_session_event("h100", 1000.0, 1000.0 + 2 * 3600)], PRICES)
    assert rep["deployment"]["hours"] == 2.0
    assert rep["deployment"]["usd"] == 14.0
    assert rep["token_usd"] == 0.0 and rep["total_usd"] == 14.0


def test_deployment_accepts_iso_timestamps_and_sums_tokens():
    events = [
        _ev("generation", "accounts/fireworks/models/gemma-3", 1_000_000, 0),  # $0.90 tokens
        deployment_session_event("b200", "2026-07-10T00:00:00Z", "2026-07-10T00:30:00Z"),  # 0.5h @ $10
    ]
    rep = report(events, PRICES)
    assert rep["deployment"]["hours"] == 0.5 and rep["deployment"]["usd"] == 5.0
    assert rep["token_usd"] == 0.9 and rep["total_usd"] == 5.9
