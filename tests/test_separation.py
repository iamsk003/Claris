"""gate_3 (tone separation) tests. The sarcastic vs humorous_non_tech collision explicit."""

from __future__ import annotations

import asyncio

from claris.core.verification.config import VerificationConfig
from claris.core.verification.separation import (
    build_contrast_prompt,
    cosine,
    enforce_separation,
    max_collision,
    pairwise_similarities,
)
from claris.core.schema import CaptionCandidate, CritiqueScore, StyleName

CFG = VerificationConfig()
S = StyleName


def _cand(style, text, cid) -> CaptionCandidate:
    return CaptionCandidate(candidate_id=cid, style=style, text=text,
                            evidence_ids=("E001",), temperature=0.3, seed=1, model="fake")


def _score(overall) -> CritiqueScore:
    return CritiqueScore(accuracy=overall, tone_fidelity=overall, style_distinctness=overall,
                         naturalness=overall, overall=overall)


def test_cosine_basics():
    assert cosine([1, 0], [1, 0]) == 1.0
    assert abs(cosine([1, 0], [0, 1])) < 1e-9
    assert cosine([0, 0], [1, 1]) == 0.0


def test_pairwise_has_six_pairs():
    styles = [S.FORMAL, S.SARCASTIC, S.HUMOROUS_TECH, S.HUMOROUS_NON_TECH]
    vecs = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    sims = pairwise_similarities(styles, vecs)
    assert len(sims) == 6 and max(sims.values()) == 0.0
    assert max_collision(sims, 0.82) is None


def test_build_contrast_prompt_quotes_other():
    p = build_contrast_prompt(S.HUMOROUS_NON_TECH, S.SARCASTIC, "already said this")
    assert "already said this" in p and "sarcastic" in p and "non_tech" in p


def _embed_by_text(vectors: dict[str, list[float]]):
    # Deterministic embedder: exact-match text -> vector.
    return lambda texts: [vectors[t] for t in texts]


def test_sarcastic_vs_non_tech_collision_is_resolved():
    # sarcastic and humorous_non_tech start nearly identical; non_tech scores lower and is
    # the one regenerated. After regeneration its embedding is distinct and the pair clears.
    collide = [1.0, 1.0, 0.0, 0.0]
    sarc_text = "Oh good, another dog catching a frisbee. Thrilling."
    nt_text = "Oh good, another dog catching a frisbee. Thrilling stuff."
    new_nt_text = "That dog treats a frisbee like the single most important job on earth."

    selected = {
        S.FORMAL: (_cand(S.FORMAL, "A dog catches a frisbee.", "f"), _score(4.0)),
        S.SARCASTIC: (_cand(S.SARCASTIC, sarc_text, "s"), _score(4.2)),
        S.HUMOROUS_TECH: (_cand(S.HUMOROUS_TECH, "Frisbee retrieval: zero downtime.", "ht"), _score(4.0)),
        S.HUMOROUS_NON_TECH: (_cand(S.HUMOROUS_NON_TECH, nt_text, "nt"), _score(3.5)),
    }
    embed = _embed_by_text({
        "A dog catches a frisbee.": [1.0, 0.0, 0.0, 0.0],
        sarc_text: collide,
        "Frisbee retrieval: zero downtime.": [0.0, 0.0, 1.0, 0.0],
        nt_text: collide,                        # collides with sarcastic
        new_nt_text: [0.0, 0.0, 0.0, 1.0],       # orthogonal after regeneration
    })

    regenerated: list[StyleName] = []

    async def replace_fn(style, contrast):
        regenerated.append(style)
        assert sarc_text in contrast  # contrast quotes the sarcastic caption
        return (_cand(style, new_nt_text, "nt2"), _score(3.8))

    out = asyncio.run(
        enforce_separation(selected, CFG, replace_fn=replace_fn, embed_fn=embed)
    )
    # The lower-scoring colliding style (non_tech) was regenerated, once.
    assert regenerated == [S.HUMOROUS_NON_TECH]
    assert out[S.HUMOROUS_NON_TECH][0].text == new_nt_text
    # Final captions are all below threshold.
    styles = list(out.keys())
    sims = pairwise_similarities(styles, embed([out[s][0].text for s in styles]))
    assert max_collision(sims, CFG.separation_threshold) is None


def test_no_collision_no_regeneration():
    selected = {
        S.FORMAL: (_cand(S.FORMAL, "a", "f"), _score(4.0)),
        S.SARCASTIC: (_cand(S.SARCASTIC, "b", "s"), _score(4.0)),
        S.HUMOROUS_TECH: (_cand(S.HUMOROUS_TECH, "c", "ht"), _score(4.0)),
        S.HUMOROUS_NON_TECH: (_cand(S.HUMOROUS_NON_TECH, "d", "nt"), _score(4.0)),
    }
    embed = _embed_by_text({"a": [1, 0, 0, 0], "b": [0, 1, 0, 0], "c": [0, 0, 1, 0], "d": [0, 0, 0, 1]})

    async def replace_fn(style, contrast):
        raise AssertionError("should not regenerate when there is no collision")

    out = asyncio.run(enforce_separation(selected, CFG, replace_fn=replace_fn, embed_fn=embed))
    assert out == selected


def test_gives_up_after_max_rounds():
    # Persistent collision the replacement never fixes: must stop after max_rounds, not loop.
    collide = [1.0, 1.0]
    selected = {
        S.SARCASTIC: (_cand(S.SARCASTIC, "x", "s"), _score(4.5)),
        S.HUMOROUS_NON_TECH: (_cand(S.HUMOROUS_NON_TECH, "y", "nt"), _score(3.0)),
    }
    embed = lambda texts: [collide for _ in texts]  # everything always collides
    calls = {"n": 0}

    async def replace_fn(style, contrast):
        calls["n"] += 1
        return (_cand(style, "z", f"z{calls['n']}"), _score(3.0))

    asyncio.run(enforce_separation(selected, CFG, replace_fn=replace_fn, embed_fn=embed))
    assert calls["n"] == CFG.separation_max_rounds  # bounded, no infinite loop
