"""Tunable knobs for the verification gates. Weights and thresholds live here, not
as constants scattered through the gate code."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CriticWeights:
    """Weights for the 1-5 critic dimensions. Must sum to 1.0."""

    accuracy: float = 0.40
    tone_fidelity: float = 0.35
    style_distinctness: float = 0.15
    naturalness: float = 0.10


@dataclass(frozen=True)
class VerificationConfig:
    """Defaults are CHEAP mode: single-seed critic, small-Gemma gate_1 judge."""

    # gate_1 — grounding / entailment.
    min_support: float = 0.8              # survive at >= this fraction of claims entailed
    hedge_penalty: float = 0.05           # subtracted from support per unused cited ID

    # gate_2 — critic. Cheap mode uses one seed; thorough() averages two.
    weights: CriticWeights = field(default_factory=CriticWeights)
    critic_seeds: tuple[int, ...] = (101,)
    critic_temperature: float = 0.3

    # gate_3 — tone separation.
    separation_threshold: float = 0.82
    separation_max_rounds: int = 2

    # Models. gate_1 only needs parseable JSON, so it runs the smallest reliable Gemma;
    # gate_2 selection quality matters more, so its critic stays on the larger Gemma.
    judge_model: str = "accounts/fireworks/models/gemma-3-4b"
    critic_model: str = "accounts/fireworks/models/gemma-3"
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Timeouts (seconds).
    judge_timeout_s: float = 60.0
    critic_timeout_s: float = 60.0

    @classmethod
    def thorough(cls) -> "VerificationConfig":
        """Two-seed critic averaging. Use once, for the final table."""
        return cls(critic_seeds=(101, 202))
