"""Cost meter. Reads the structured JSONL event log and reports tokens + estimated USD.

Every billable call is logged as an ``llm_call`` RunEvent carrying ``tokens_in``,
``tokens_out``, ``model``, and a ``cost_stage`` (perception_vision, generation, gate_1,
gate_2, gate_3, held_out_judge). This reads those events, groups by stage and by model, and
prices them against ``eval/fireworks_prices.json`` (real Fireworks serverless rates; see
that file for source and the per-model bucket assumptions). No price is hardcoded here.

    python -m eval.spend .claris_logs/run.jsonl
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

PRICES_PATH = Path(__file__).parent / "fireworks_prices.json"


def _to_epoch(value) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()


def deployment_rate(gpu: str, prices: dict) -> float:
    return prices.get("deployment_gpu_hourly", {}).get((gpu or "").lower(), 0.0)


def deployment_session_event(gpu: str, start, stop) -> dict:
    """Build a deployment_session event the meter can price. start/stop: epoch or ISO."""
    return {"event_type": "deployment_session",
            "payload": {"gpu": gpu, "start": start, "stop": stop}}


def load_prices(path: Optional[Path] = None) -> dict:
    return json.loads((path or PRICES_PATH).read_text())


def price_per_1m(model: Optional[str], prices: dict) -> float:
    if model and model in prices.get("model_prices_per_1m", {}):
        return prices["model_prices_per_1m"][model]
    return prices.get("default_price_per_1m", 0.90)


def read_events(path: str | Path) -> list[dict]:
    lines = Path(path).read_text().splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def report(events: Iterable[dict], prices: dict) -> dict:
    """Aggregate llm_call events into a spend report. Pure over the given events."""
    by_stage: dict[str, dict[str, float]] = defaultdict(lambda: {"tokens": 0.0, "usd": 0.0, "calls": 0.0})
    by_model: dict[str, dict[str, float]] = defaultdict(lambda: {"tokens": 0.0, "usd": 0.0, "calls": 0.0})
    total_tokens = 0.0
    total_usd = 0.0

    deploy = {"hours": 0.0, "usd": 0.0, "sessions": 0}
    for ev in events:
        et = ev.get("event_type")
        if et == "deployment_session":
            p = ev.get("payload") or {}
            hours = max(0.0, (_to_epoch(p["stop"]) - _to_epoch(p["start"])) / 3600.0)
            deploy["hours"] += hours
            deploy["usd"] += hours * deployment_rate(p.get("gpu", ""), prices)
            deploy["sessions"] += 1
            continue
        if et != "llm_call":
            continue
        stage = (ev.get("payload") or {}).get("cost_stage") or ev.get("stage") or "unknown"
        model = ev.get("model") or "unknown"
        tokens = (ev.get("tokens_in") or 0) + (ev.get("tokens_out") or 0)
        usd = tokens / 1_000_000 * price_per_1m(model, prices)
        for bucket, key in ((by_stage, stage), (by_model, model)):
            bucket[key]["tokens"] += tokens
            bucket[key]["usd"] += usd
            bucket[key]["calls"] += 1
        total_tokens += tokens
        total_usd += usd

    def _round(d):
        return {k: {"calls": int(v["calls"]), "tokens": int(v["tokens"]), "usd": round(v["usd"], 6)}
                for k, v in d.items()}

    return {
        "by_stage": _round(by_stage),
        "by_model": _round(by_model),
        "token_usd": round(total_usd, 6),
        "total_tokens": int(total_tokens),
        "deployment": {"hours": round(deploy["hours"], 4), "usd": round(deploy["usd"], 6),
                       "sessions": deploy["sessions"]},
        "total_usd": round(total_usd + deploy["usd"], 6),
    }


def format_report(rep: dict, prices: dict) -> str:
    rows = ["", "Spend — tokens and estimated USD", "=" * 52,
            f"{'stage':<20}{'calls':>7}{'tokens':>12}{'USD':>12}"]
    for stage in sorted(rep["by_stage"], key=lambda s: -rep["by_stage"][s]["usd"]):
        v = rep["by_stage"][stage]
        rows.append(f"{stage:<20}{v['calls']:>7}{v['tokens']:>12,}{v['usd']:>12.4f}")
    rows.append("-" * 52)
    rows.append(f"{'by model':<20}")
    for model in sorted(rep["by_model"], key=lambda m: -rep["by_model"][m]["usd"]):
        v = rep["by_model"][model]
        short = model.split("/")[-1]
        rate = price_per_1m(model, prices)
        rows.append(f"  {short:<28}{v['tokens']:>10,} tok  ${v['usd']:>9.4f}  (@${rate}/1M)")
    rows.append("-" * 52)
    rows.append(f"{'token spend':<20}{'':>7}{rep['total_tokens']:>12,}{rep.get('token_usd', 0.0):>12.4f}")
    dep = rep.get("deployment", {"hours": 0.0, "usd": 0.0, "sessions": 0})
    rows.append(f"{'deployment (GPU-hr)':<20}{'':>7}{dep['hours']:>11.3f}h{dep['usd']:>12.4f}")
    if dep["sessions"]:
        rows.append("  NOTE: dedicated deployment bills by UPTIME — idle time bills the same "
                    "as busy. Stop the deployment when not measuring.")
    rows.append("=" * 52)
    rows.append(f"{'GRAND TOTAL USD':<20}{'':>7}{'':>12}{rep['total_usd']:>12.4f}")
    rows.append(f"(token prices: {prices.get('source')}, fetched {prices.get('fetched_at')})")
    return "\n".join(rows)


def main(argv: list[str]) -> int:
    if not argv:
        print("usage: python -m eval.spend <event-log.jsonl>")
        return 2
    prices = load_prices()
    rep = report(read_events(argv[0]), prices)
    print(format_report(rep, prices))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
