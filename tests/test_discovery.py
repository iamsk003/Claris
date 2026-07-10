"""Capability discovery + role resolution. Pure logic, synthetic catalog, no network."""

from __future__ import annotations

import asyncio

from claris.agent.discovery import ModelInfo, discover, is_gemma, resolve_roles, size_hint

FW = "accounts/fireworks/models/"


def _mi(slug, chat=True, json_ok=True, vision=False):
    return ModelInfo(id=FW + slug, chat=chat, json_ok=json_ok, vision=vision,
                     size_hint=size_hint(slug))


def test_size_hint_parses_largest_b_token():
    assert size_hint("gemma-4-26b-a4b-it") == 26  # 26b, not the a4b
    assert size_hint("gpt-oss-120b") == 120
    assert size_hint("gemma-3-4b-it") == 4
    assert size_hint("glm-5p2") == 0 and size_hint("kimi-k2p6") == 0


def test_is_gemma():
    assert is_gemma(FW + "gemma-4-26b-a4b-it") and not is_gemma(FW + "gpt-oss-120b")


def test_resolve_prefers_gemma_gen_strongest_critic_smallest_json_gate1():
    models = [
        _mi("gemma-4-26b-a4b-it", vision=True),  # gemma + vision
        _mi("gpt-oss-120b"),                     # strongest chat overall
        _mi("gemma-3-4b-it"),                    # smallest json-capable
    ]
    r = resolve_roles(models)
    assert r.vlm.endswith("gemma-4-26b-a4b-it")     # gemma with vision
    assert r.gen.endswith("gemma-4-26b-a4b-it")     # strongest gemma
    assert r.critic.endswith("gpt-oss-120b")        # strongest chat, gemma or not
    assert r.gate1.endswith("gemma-3-4b-it")        # smallest json-capable
    assert r.gemma_path_used is True


def test_resolve_no_gemma_no_vision_degrades():
    models = [_mi("gpt-oss-120b"), _mi("glm-5p2")]
    r = resolve_roles(models)
    assert r.vlm is None                       # no vision model -> perception degrades
    assert r.gemma_path_used is False          # generation not on Gemma
    assert r.gen.endswith("gpt-oss-120b")      # strongest chat still assigned
    assert r.gate1 is not None
    assert any("gemma_path_used=False" in n for n in r.notes)


def test_resolve_vision_fallback_non_gemma():
    # kimi accepts images -> becomes the VLM even though it is not Gemma.
    models = [_mi("kimi-k2p6", vision=True), _mi("gpt-oss-120b")]
    r = resolve_roles(models)
    assert r.vlm.endswith("kimi-k2p6")
    assert any("no Gemma vision" in n for n in r.notes)


def test_known_good_vision_preferred_over_other_fallback():
    models = [_mi("kimi-k2p6", vision=True), _mi("qwen3p7-plus", vision=True), _mi("gpt-oss-120b")]
    r = resolve_roles(models)
    assert r.vlm.endswith("qwen3p7-plus")   # known-good vision beats other reachable vision
    assert r.gen.endswith("gpt-oss-120b")   # known-good chat fallback


def test_discover_uses_injected_probes():
    async def list_fn():
        return [FW + "a-7b", FW + "b-1b"]

    async def probe(mid):
        return (True, True, "7b" in mid)  # only the 7b model is vision-capable

    infos = asyncio.run(discover(list_fn, probe, extra_candidates=()))
    assert len(infos) == 2
    assert infos[0].vision and not infos[1].vision
    assert infos[0].size_hint == 7 and infos[1].size_hint == 1


def test_discover_probes_known_good_even_if_unlisted():
    async def list_fn():
        return [FW + "glm-5p2"]

    async def probe(mid):
        return (True, True, "qwen" in mid)  # qwen (known-good) is vision-capable

    ids = [m.id for m in asyncio.run(discover(list_fn, probe))]
    assert FW + "qwen3p7-plus" in ids and FW + "gpt-oss-120b" in ids
