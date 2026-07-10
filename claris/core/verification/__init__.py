"""Verification — the three gates.

gate_1 (hallucination.py): grounding/entailment. Eliminates contradicted or
under-supported candidates; penalizes hedging; regenerates then degrades so a style is
never empty.
gate_2 (critic.py): a Gemma critic scores each survivor 1-5 on four config-weighted
dimensions and selects the argmax.
gate_3 (separation.py): embeds the four winners and pulls apart any pair that collides.
"""

from claris.core.verification.config import CriticWeights, VerificationConfig
from claris.core.verification.critic import (
    critique_candidate,
    score_and_select,
    weighted_overall,
)
from claris.core.verification.hallucination import (
    ClaimAssessment,
    ClaimVerdict,
    Gate1Outcome,
    GroundingReport,
    assess_candidate,
    evaluate_grounding,
    gate_1,
    ground_style,
    negative_constraint,
    support_score,
)
from claris.core.verification.separation import (
    enforce_separation,
    max_collision,
    mean_pairwise_similarity,
    pairwise_similarities,
)

__all__ = [
    "VerificationConfig",
    "CriticWeights",
    # gate_1
    "ClaimAssessment",
    "ClaimVerdict",
    "GroundingReport",
    "Gate1Outcome",
    "support_score",
    "evaluate_grounding",
    "gate_1",
    "ground_style",
    "negative_constraint",
    "assess_candidate",
    # gate_2
    "weighted_overall",
    "critique_candidate",
    "score_and_select",
    # gate_3
    "enforce_separation",
    "pairwise_similarities",
    "max_collision",
    "mean_pairwise_similarity",
]
