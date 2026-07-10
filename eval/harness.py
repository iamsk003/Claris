"""Offline evaluation harness + ablation table.

Runs the full pipeline over the golden set with gates ON and gates OFF, and scores both
arms with the HELD-OUT judge in ``eval/judge.py`` — a separate model and rubric that never
drives selection. The selection critic (gate_2) keeps selecting; it no longer measures.
The table reports mean judge score per style plus mean pairwise separation, and breaks the
delta into accuracy vs tone so we can say what the gates actually buy.

    python -m eval.harness                       # full ablation, writes eval/baseline.json
    python -m eval.harness --no-gates            # gates-off arm only
    python -m eval.harness --providers fireworks # real numbers (needs FIREWORKS_API_KEY)
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import statistics
import time
from pathlib import Path
from typing import Optional

from claris.core.generation import GenerationConfig, StyleContractRegistry
from claris.core.observability import EventSink, JSONLSink, NullSink
from claris.core.pipeline import Providers, run_from_ledger
from claris.core.providers.base import ChatProvider
from claris.core.schema import ALL_STYLES, EvidenceLedger, StyleName, Task, TaskResult
from claris.core.verification import VerificationConfig
from claris.core.verification.separation import mean_pairwise_similarity, pairwise_similarities
from eval import spend
from eval.judge import DEFAULT_EVAL_JUDGE_MODEL, score_caption

GOLDEN_DIR = Path(__file__).parent / "golden"
BASELINE_PATH = Path(__file__).parent / "baseline.json"
_DIMS = ("accuracy", "tone_fidelity", "overall")


def load_golden_ledgers() -> list[EvidenceLedger]:
    """Load and validate every fixture ledger in eval/golden/."""
    ledgers: list[EvidenceLedger] = []
    for path in sorted(GOLDEN_DIR.glob("ledger_*.json")):
        ledgers.append(EvidenceLedger.model_validate(json.loads(path.read_text())))
    return ledgers


async def _measure(
    result: TaskResult, ledger: EvidenceLedger, judge: ChatProvider,
    registry: StyleContractRegistry, judge_model: Optional[str],
    *, sink: Optional[EventSink] = None, run_id: str = "eval",
) -> dict[str, dict[str, float]]:
    """Score each final caption with the held-out judge. Returns {style: {dim: score}}."""
    out: dict[str, dict[str, float]] = {}
    for style, cap in result.captions.items():
        js = await score_caption(cap.text, style, registry.get(style), ledger, judge,
                                 model=judge_model, sink=sink, run_id=run_id)
        out[style.value] = {"accuracy": js.accuracy, "tone_fidelity": js.tone_fidelity,
                            "overall": js.overall}
    return out


def _separation(result: TaskResult, providers: Providers) -> tuple[float, float]:
    styles = list(result.captions)
    embed = providers.embed_fn
    assert embed is not None, "measurement needs an embed_fn (mock or MiniLM)"
    vecs = embed([result.captions[s].text for s in styles])
    sims = pairwise_similarities(styles, vecs)
    max_sim = max(sims.values()) if sims else 0.0
    return round(1.0 - mean_pairwise_similarity(sims), 4), round(max_sim, 4)


async def run_ablation(
    ledgers: list[EvidenceLedger], providers: Providers, judge: ChatProvider,
    *, judge_model: Optional[str] = None, cfg: Optional[VerificationConfig] = None,
    gen_config: Optional[GenerationConfig] = None, sink: Optional[EventSink] = None,
) -> dict:
    cfg = cfg or VerificationConfig()
    sink = sink or NullSink()
    registry = StyleContractRegistry()
    # arm -> dim -> style -> [per-clip scores]
    acc = {"on": {d: {s.value: [] for s in ALL_STYLES} for d in _DIMS},
           "off": {d: {s.value: [] for s in ALL_STYLES} for d in _DIMS}}
    sep = {"on": [], "off": []}
    maxsim = {"on": [], "off": []}

    for ledger in ledgers:
        task = Task(task_id=ledger.task_id, video_path=f"{ledger.task_id}.mp4", styles=ALL_STYLES)
        results = {
            "on": await run_from_ledger(ledger, task, providers, gen_config=gen_config,
                                        ver_config=cfg, registry=registry, sink=sink,
                                        use_gates=True, run_id=f"{ledger.task_id}_on"),
            "off": await run_from_ledger(ledger, task, providers, gen_config=gen_config,
                                         ver_config=cfg, registry=registry, sink=sink,
                                         use_gates=False, run_id=f"{ledger.task_id}_off"),
        }
        for arm, result in results.items():
            measured = await _measure(result, ledger, judge, registry, judge_model,
                                      sink=sink, run_id=f"{ledger.task_id}_{arm}")
            for s in ALL_STYLES:
                for d in _DIMS:
                    acc[arm][d][s.value].append(measured[s.value][d])
            s_, m_ = _separation(result, providers)
            sep[arm].append(s_)
            maxsim[arm].append(m_)

    def mean(xs):
        return round(statistics.mean(xs), 4) if xs else 0.0

    def per_style(arm, dim):
        return {s.value: mean(acc[arm][dim][s.value]) for s in ALL_STYLES}

    gates_on = per_style("on", "overall")
    gates_off = per_style("off", "overall")
    delta = {s: round(gates_on[s] - gates_off[s], 4) for s in gates_on}

    def dim_mean(arm, dim):
        return round(statistics.mean([v for s in ALL_STYLES for v in acc[arm][dim][s.value]]), 4)

    acc_delta = round(dim_mean("on", "accuracy") - dim_mean("off", "accuracy"), 4)
    tone_delta = round(dim_mean("on", "tone_fidelity") - dim_mean("off", "tone_fidelity"), 4)
    overall_delta = round(dim_mean("on", "overall") - dim_mean("off", "overall"), 4)
    if overall_delta <= 0:
        driver = "none"
    else:
        driver = "accuracy" if acc_delta >= tone_delta else "tone_fidelity"

    return {
        "judge_model": judge_model or getattr(judge, "model", "held-out-judge"),
        "gates_on": gates_on, "gates_off": gates_off, "delta": delta,
        "mean_score_on": dim_mean("on", "overall"), "mean_score_off": dim_mean("off", "overall"),
        "mean_delta": overall_delta,
        "accuracy_on": dim_mean("on", "accuracy"), "accuracy_off": dim_mean("off", "accuracy"),
        "accuracy_delta": acc_delta,
        "tone_on": dim_mean("on", "tone_fidelity"), "tone_off": dim_mean("off", "tone_fidelity"),
        "tone_delta": tone_delta,
        "driver": driver,
        "mean_separation_on": mean(sep["on"]), "mean_separation_off": mean(sep["off"]),
        "max_similarity_on": mean(maxsim["on"]), "max_similarity_off": mean(maxsim["off"]),
        "separation_threshold": cfg.separation_threshold,
        "n_clips": len(ledgers),
    }


def _pair_sim(result: TaskResult, providers: Providers) -> tuple[list[str], float, float]:
    """Return (worst_pair, worst_sim, sarcastic_vs_non_tech_sim) for one caption set."""
    styles = list(result.captions)
    vecs = providers.embed_fn([result.captions[s].text for s in styles])
    sims = pairwise_similarities(styles, vecs)
    (pa, pb), mx = max(sims.items(), key=lambda kv: kv[1])
    key = (StyleName.SARCASTIC, StyleName.HUMOROUS_NON_TECH)
    target = sims.get(key)
    if target is None:
        target = sims.get((StyleName.HUMOROUS_NON_TECH, StyleName.SARCASTIC), 0.0)
    return [pa.value, pb.value], round(mx, 4), round(target, 4)


async def run_three_arm(
    ledgers: list[EvidenceLedger], providers: Providers, judge: ChatProvider,
    *, judge_model: Optional[str] = None, cfg: Optional[VerificationConfig] = None,
    gen_base: Optional[GenerationConfig] = None, sink: Optional[EventSink] = None,
) -> dict:
    """A: gates off / parallel. B: gates on / parallel. C: gates on / sequential."""
    cfg = cfg or VerificationConfig()
    sink = sink or NullSink()
    gen_base = gen_base or GenerationConfig()
    registry = StyleContractRegistry()
    arms = {
        "A_off_parallel": dict(use_gates=False, sequential=False),
        "B_on_parallel": dict(use_gates=True, sequential=False),
        "C_on_sequential": dict(use_gates=True, sequential=True),
    }
    acc = {a: {d: [] for d in _DIMS} for a in arms}
    wall = {a: [] for a in arms}
    per_clip = []

    for ledger in ledgers:
        task = Task(task_id=ledger.task_id, video_path=f"{ledger.task_id}.mp4", styles=ALL_STYLES)
        for arm, opts in arms.items():
            gen_config = dataclasses.replace(gen_base, sequential=opts["sequential"])
            t0 = time.perf_counter()
            result = await run_from_ledger(ledger, task, providers, gen_config=gen_config,
                                           ver_config=cfg, registry=registry, sink=sink,
                                           use_gates=opts["use_gates"], run_id=f"{ledger.task_id}_{arm}")
            wall[arm].append(time.perf_counter() - t0)
            measured = await _measure(result, ledger, judge, registry, judge_model,
                                      sink=sink, run_id=f"{ledger.task_id}_{arm}")
            for d in _DIMS:
                acc[arm][d].append(statistics.mean(measured[s.value][d] for s in ALL_STYLES))
            worst_pair, worst_sim, target = _pair_sim(result, providers)
            per_clip.append({"clip": ledger.task_id, "arm": arm, "worst_pair": worst_pair,
                             "worst_sim": worst_sim, "sarcastic_vs_non_tech": target})

    def m(xs):
        return round(statistics.mean(xs), 4) if xs else 0.0

    per_arm = {a: {d: m(acc[a][d]) for d in _DIMS} | {"wall_clock_s": m(wall[a])} for a in arms}
    base = per_arm["A_off_parallel"]
    deltas = {
        a: {"accuracy": round(per_arm[a]["accuracy"] - base["accuracy"], 4),
            "tone_fidelity": round(per_arm[a]["tone_fidelity"] - base["tone_fidelity"], 4),
            "overall": round(per_arm[a]["overall"] - base["overall"], 4)}
        for a in ("B_on_parallel", "C_on_sequential")
    }
    return {"per_arm": per_arm, "deltas_vs_A": deltas, "per_clip_similarity": per_clip,
            "separation_threshold": cfg.separation_threshold, "n_clips": len(ledgers),
            "judge_model": judge_model or getattr(judge, "model", "held-out-judge")}


def _format_three_arm(t: dict) -> str:
    rows = ["", f"Three-arm — held-out judge ({t['judge_model']})", "-" * 70,
            f"{'arm':<20}{'accuracy':>10}{'tone':>8}{'overall':>9}{'wall_s':>9}"]
    for a in ("A_off_parallel", "B_on_parallel", "C_on_sequential"):
        v = t["per_arm"][a]
        rows.append(f"{a:<20}{v['accuracy']:>10.3f}{v['tone_fidelity']:>8.3f}"
                    f"{v['overall']:>9.3f}{v['wall_clock_s']:>9.2f}")
    rows.append("-" * 70)
    for a, d in t["deltas_vs_A"].items():
        rows.append(f"{a} vs A: accuracy {d['accuracy']:+.3f}, tone {d['tone_fidelity']:+.3f}, "
                    f"overall {d['overall']:+.3f}")
    rows.append(f"\nmax pairwise similarity per clip (threshold {t['separation_threshold']}):")
    for r in t["per_clip_similarity"]:
        rows.append(f"  {r['clip']:<20} {r['arm']:<18} worst {r['worst_sim']:.3f} "
                    f"{r['worst_pair']}  sarc/non_tech {r['sarcastic_vs_non_tech']:.3f}")
    return "\n".join(rows)


def _format_table(t: dict) -> str:
    rows = ["", f"Ablation — held-out judge ({t['judge_model']}), mean score per style (1-5)",
            "-" * 58, f"{'style':<20}{'gates off':>12}{'gates on':>12}{'delta':>12}"]
    for style in ALL_STYLES:
        s = style.value
        rows.append(f"{s:<20}{t['gates_off'][s]:>12.3f}{t['gates_on'][s]:>12.3f}{t['delta'][s]:>+12.3f}")
    rows.append("-" * 58)
    rows.append(f"{'OVERALL':<20}{t['mean_score_off']:>12.3f}{t['mean_score_on']:>12.3f}{t['mean_delta']:>+12.3f}")
    rows.append(f"{'  accuracy':<20}{t['accuracy_off']:>12.3f}{t['accuracy_on']:>12.3f}{t['accuracy_delta']:>+12.3f}")
    rows.append(f"{'  tone_fidelity':<20}{t['tone_off']:>12.3f}{t['tone_on']:>12.3f}{t['tone_delta']:>+12.3f}")
    rows.append(f"{'max similarity':<20}{t['max_similarity_off']:>12.3f}{t['max_similarity_on']:>12.3f}"
                f"   (threshold {t['separation_threshold']})")
    return "\n".join(rows)


def build_providers(kind: str) -> Providers:
    if kind == "mock":
        from eval.mock_providers import mock_providers
        return mock_providers()
    if kind == "fireworks":
        from claris.core.providers.fireworks import FireworksProvider
        from claris.core.verification.config import VerificationConfig as _VC
        from claris.core.verification.separation import _minilm_embed
        cfg = _VC()
        return Providers(
            gen_provider=FireworksProvider(), judge=FireworksProvider(model=cfg.judge_model),
            critic=FireworksProvider(model=cfg.critic_model),
            embed_fn=lambda texts: _minilm_embed(texts, cfg),
        )
    raise ValueError(f"unknown providers kind: {kind}")


def build_eval_judge(kind: str) -> tuple[ChatProvider, Optional[str]]:
    """The held-out judge provider + model. Distinct from gate_2's critic."""
    if kind == "mock":
        from eval.mock_providers import MockHeldOutJudge
        return MockHeldOutJudge(), None
    if kind == "fireworks":
        from claris.core.providers.fireworks import FireworksProvider
        return FireworksProvider(model=DEFAULT_EVAL_JUDGE_MODEL), DEFAULT_EVAL_JUDGE_MODEL
    raise ValueError(f"unknown providers kind: {kind}")


def main() -> int:
    ap = argparse.ArgumentParser(description="CLARIS eval harness / ablation table")
    ap.add_argument("--no-gates", action="store_true", help="run the gates-off arm only")
    ap.add_argument("--providers", default="mock", choices=["mock", "fireworks"])
    ap.add_argument("--thorough", action="store_true",
                    help="N=4 + two-seed critic. Use once, for the final table.")
    ap.add_argument("--three-arm", action="store_true",
                    help="A: gates off/parallel, B: gates on/parallel, C: gates on/sequential.")
    ap.add_argument("--out", default=str(BASELINE_PATH))
    args = ap.parse_args()

    ledgers = load_golden_ledgers()
    providers = build_providers(args.providers)
    judge, judge_model = build_eval_judge(args.providers)
    gen_config = GenerationConfig.thorough() if args.thorough else GenerationConfig()
    cfg = VerificationConfig.thorough() if args.thorough else VerificationConfig()
    profile = "thorough" if args.thorough else "cheap"
    registry = StyleContractRegistry()

    log_path = Path(".claris_logs") / f"harness_{args.providers}_{profile}.jsonl"
    if log_path.exists():
        log_path.unlink()
    sink = JSONLSink(log_path)

    def _print_spend() -> dict:
        prices = spend.load_prices()
        rep = spend.report(spend.read_events(log_path) if log_path.exists() else [], prices)
        print(spend.format_report(rep, prices))
        return rep

    if args.three_arm:
        table = asyncio.run(run_three_arm(ledgers, providers, judge, judge_model=judge_model,
                                          cfg=cfg, gen_base=gen_config, sink=sink))
        rep = _print_spend()
        out = Path(args.out).with_name("three_arm.json")
        out.write_text(json.dumps({"synthetic": args.providers == "mock", "profile": profile,
                                   "spend_usd": rep["total_usd"], **table}, indent=2) + "\n")
        print(_format_three_arm(table))
        print(f"\nwrote {out}")
        d = table["deltas_vs_A"]
        print(f"\ntone_fidelity delta — B (parallel): {d['B_on_parallel']['tone_fidelity']:+.3f}, "
              f"C (sequential): {d['C_on_sequential']['tone_fidelity']:+.3f}")
        return 0

    if args.no_gates:
        async def _off():
            scores = {s.value: [] for s in ALL_STYLES}
            for ledger in ledgers:
                task = Task(task_id=ledger.task_id, video_path=f"{ledger.task_id}.mp4",
                            styles=ALL_STYLES)
                r = await run_from_ledger(ledger, task, providers, gen_config=gen_config,
                                          ver_config=cfg, registry=registry, sink=sink,
                                          use_gates=False, run_id=f"{ledger.task_id}_off")
                m = await _measure(r, ledger, judge, registry, judge_model, sink=sink,
                                   run_id=f"{ledger.task_id}_off")
                for s in ALL_STYLES:
                    scores[s.value].append(m[s.value]["overall"])
            return {s: round(statistics.mean(v), 4) for s, v in scores.items()}

        print("gates OFF, held-out judge, mean score per style:")
        print(json.dumps(asyncio.run(_off()), indent=2))
        _print_spend()
        return 0

    table = asyncio.run(run_ablation(ledgers, providers, judge, judge_model=judge_model,
                                     cfg=cfg, gen_config=gen_config, sink=sink))
    rep = _print_spend()
    doc = {
        "synthetic": args.providers == "mock",
        "providers": args.providers,
        "profile": profile,
        "measurement": "held-out judge, independent of the gate_2 selection critic",
        "note": (
            "Deterministic MOCK run (no network). The held-out judge shares no scoring "
            "signal with the selection critic; the delta reflects grounding. Regenerate "
            "with --providers fireworks over real clips for real numbers."
            if args.providers == "mock" else "Live Fireworks run over real clips."
        ),
        "spend_usd": rep["total_usd"],
        "spend_by_stage": {k: v["usd"] for k, v in rep["by_stage"].items()},
        **table,
    }
    Path(args.out).write_text(json.dumps(doc, indent=2) + "\n")
    print(_format_table(table))
    print(f"\nwrote {args.out}")
    if table["mean_delta"] <= 0:
        print("\nFINDING: under a held-out judge the gates do NOT improve the score "
              f"({table['mean_score_off']:.3f} -> {table['mean_score_on']:.3f}). "
              "By the brief, that is the result — reported, not tuned away.")
        return 1
    print(f"\nGates improve the held-out score by {table['mean_delta']:+.3f} "
          f"({table['mean_score_off']:.3f} -> {table['mean_score_on']:.3f}), "
          f"driven by {table['driver']} "
          f"(accuracy {table['accuracy_delta']:+.3f}, tone {table['tone_delta']:+.3f}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
