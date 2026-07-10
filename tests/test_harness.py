"""The ablation table is the deliverable: guard that the gates help under a HELD-OUT judge.

The judge in these tests shares no scoring signal with the selection critic, so a positive
delta means the gates improved grounding, not that the critic graded its own paper.
"""

from __future__ import annotations

import asyncio

from claris.core.verification import VerificationConfig
from eval.harness import load_golden_ledgers, run_ablation, run_three_arm
from eval.mock_providers import MockHeldOutJudge, mock_providers


def _table():
    return asyncio.run(
        run_ablation(load_golden_ledgers()[:1], mock_providers(), MockHeldOutJudge())
    )


def test_gates_improve_under_held_out_judge():
    table = _table()
    assert table["mean_delta"] > 0
    assert table["mean_score_on"] > table["mean_score_off"]


def test_delta_is_attributed_to_a_dimension():
    table = _table()
    # The mock judge's accuracy is grounding-driven and its tone is style-agnostic, so the
    # gate win must come from accuracy, not tone.
    assert table["driver"] == "accuracy"
    assert table["accuracy_delta"] > 0


def test_gate_3_keeps_styles_under_collision_threshold():
    table = _table()
    assert table["max_similarity_on"] <= VerificationConfig().separation_threshold


def test_three_arm_structure_and_named_pairs():
    table = asyncio.run(
        run_three_arm(load_golden_ledgers()[:1], mock_providers(), MockHeldOutJudge())
    )
    assert set(table["per_arm"]) == {"A_off_parallel", "B_on_parallel", "C_on_sequential"}
    assert set(table["deltas_vs_A"]) == {"B_on_parallel", "C_on_sequential"}
    # Gates on (B) improve accuracy over gates off (A) under the independent judge.
    assert table["deltas_vs_A"]["B_on_parallel"]["accuracy"] > 0
    # Every per-clip row names the worst pair and reports the sarcastic/non_tech pair.
    for r in table["per_clip_similarity"]:
        assert len(r["worst_pair"]) == 2 and "sarcastic_vs_non_tech" in r
    # Every arm records wall-clock.
    assert all("wall_clock_s" in v for v in table["per_arm"].values())
