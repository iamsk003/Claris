"""gate_3 — tone separation.

The four selected captions are embedded (sentence-transformers MiniLM, local, no API)
and their six pairwise cosine similarities computed. If any pair exceeds the separation
threshold, the lower-scoring caption of that pair is regenerated with a contrast prompt
that quotes what the other style already said and forbids overlap in phrasing, joke
structure, and observation. The replacement is re-grounded (gate_1) and re-scored (gate_2)
by the injected ``replace_fn``. Up to two rounds, then accept and log.

sarcastic vs humorous_non_tech is the collision that actually shows up: both can drift to
dry one-liners about the same moment. The embedding + contrast loop pulls them apart.

Embedding and cosine are pure/injectable, so this runs offline in tests.
"""

from __future__ import annotations

import math
from itertools import combinations
from typing import Awaitable, Callable, Optional

from claris.core.observability import EventSink, NullSink
from claris.core.schema import CaptionCandidate, CritiqueScore, RunEvent, StyleName
from claris.core.verification.config import VerificationConfig

Selection = dict[StyleName, tuple[CaptionCandidate, CritiqueScore]]
ReplaceFn = Callable[[StyleName, str], Awaitable[Optional[tuple[CaptionCandidate, CritiqueScore]]]]
EmbedFn = Callable[[list[str]], list[list[float]]]


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two vectors. Pure. Returns 0.0 for a zero vector."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def pairwise_similarities(
    styles: list[StyleName], vectors: list[list[float]]
) -> dict[tuple[StyleName, StyleName], float]:
    """The six pairwise cosine similarities among the four captions. Pure."""
    sims: dict[tuple[StyleName, StyleName], float] = {}
    for i, j in combinations(range(len(styles)), 2):
        sims[(styles[i], styles[j])] = cosine(vectors[i], vectors[j])
    return sims


def max_collision(
    sims: dict[tuple[StyleName, StyleName], float], threshold: float
) -> Optional[tuple[tuple[StyleName, StyleName], float]]:
    """The worst offending pair above threshold, or None. Pure."""
    over = [(pair, s) for pair, s in sims.items() if s > threshold]
    if not over:
        return None
    return max(over, key=lambda ps: ps[1])


def build_contrast_prompt(loser: StyleName, other: StyleName, other_text: str) -> str:
    """Contrast instruction quoting what the other style already said. Pure."""
    return (
        f"Another caption, in the {other.value} style, already said:\n"
        f'  "{other_text}"\n'
        f"Write the {loser.value} caption so it does NOT overlap with that. Forbidden: "
        "reusing its phrasing, its joke structure, or its central observation. Choose a "
        "different angle, a different detail from the evidence, and different wording."
    )


def mean_pairwise_similarity(
    sims: dict[tuple[StyleName, StyleName], float]
) -> float:
    """Mean of the pairwise similarities. Pure. 0.0 if empty."""
    return sum(sims.values()) / len(sims) if sims else 0.0


async def enforce_separation(
    selected: Selection,
    cfg: Optional[VerificationConfig] = None,
    *,
    replace_fn: ReplaceFn,
    embed_fn: Optional[EmbedFn] = None,
    sink: Optional[EventSink] = None,
    run_id: str = "verification",
) -> Selection:
    """Pull colliding styles apart. Mutates a copy of ``selected`` and returns it."""
    cfg = cfg or VerificationConfig()
    sink = sink or NullSink()
    embed_fn = embed_fn or (lambda texts: _minilm_embed(texts, cfg))
    result: Selection = dict(selected)

    for round_idx in range(cfg.separation_max_rounds):
        styles = list(result.keys())
        vectors = embed_fn([result[s][0].text for s in styles])
        sims = pairwise_similarities(styles, vectors)
        collision = max_collision(sims, cfg.separation_threshold)

        sink.emit(
            RunEvent(
                run_id=run_id, event_id=f"{run_id}:separation:round{round_idx}",
                stage="verification", event_type="separation_checked",
                payload={
                    "round": round_idx,
                    "max_similarity": max(sims.values()) if sims else 0.0,
                    "mean_similarity": round(mean_pairwise_similarity(sims), 4),
                    "collision": [p.value for p in collision[0]] if collision else None,
                },
            )
        )
        if collision is None:
            return result

        (style_a, style_b), _sim = collision
        # Regenerate the lower-scoring caption of the colliding pair.
        loser = min((style_a, style_b), key=lambda s: result[s][1].overall)
        other = style_b if loser == style_a else style_a
        contrast = build_contrast_prompt(loser, other, result[other][0].text)

        replacement = await replace_fn(loser, contrast)
        if replacement is None:
            sink.emit(
                RunEvent(
                    run_id=run_id, event_id=f"{run_id}:separation:round{round_idx}:unresolved",
                    stage="verification", event_type="separation_unresolved",
                    level="warn",  # type: ignore[arg-type]
                    payload={"style": loser.value},
                )
            )
            return result  # cannot improve; accept and stop
        result[loser] = replacement

    # Ran out of rounds; accept whatever we have and log the final state.
    styles = list(result.keys())
    sims = pairwise_similarities(styles, embed_fn([result[s][0].text for s in styles]))
    sink.emit(
        RunEvent(
            run_id=run_id, event_id=f"{run_id}:separation:accepted",
            stage="verification", event_type="separation_accepted",
            level="warn" if max_collision(sims, cfg.separation_threshold) else "info",  # type: ignore[arg-type]
            payload={"max_similarity": max(sims.values()) if sims else 0.0},
        )
    )
    return result


def _minilm_embed(texts: list[str], cfg: VerificationConfig) -> list[list[float]]:  # pragma: no cover
    from sentence_transformers import SentenceTransformer  # noqa: PLC0415

    model = SentenceTransformer(cfg.embed_model)
    return [list(map(float, v)) for v in model.encode(texts, normalize_embeddings=True)]


__all__ = [
    "cosine",
    "pairwise_similarities",
    "max_collision",
    "mean_pairwise_similarity",
    "build_contrast_prompt",
    "enforce_separation",
]
