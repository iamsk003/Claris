"""Capability discovery. The container resolves models from whatever a key exposes.

At startup we GET /v1/models and probe each for chat, JSON, and vision capability, then
assign roles from what is actually reachable — never from a hardcoded slug. Gemma is
preferred wherever it exists, but nothing requires it: the shipped container must run on
the evaluator's key, where our dedicated Gemma deployment does not exist.

The probing is injectable (``list_fn`` / ``probe_fn``) so resolution is unit-tested with a
synthetic catalog and zero network. Real probes live in ``fireworks_catalog``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional

_SIZE_RE = re.compile(r"(\d+(?:\.\d+)?)b(?:\b|-)")

_FW = "accounts/fireworks/models/"
# Confirmed-working serverless fallbacks. Always probed (even if not in GET /models) and
# preferred when no Gemma is reachable for a role.
KNOWN_GOOD_CANDIDATES = (_FW + "qwen3p7-plus", _FW + "gpt-oss-120b")
KNOWN_GOOD_VISION = ("qwen3p7-plus",)
KNOWN_GOOD_CHAT = ("gpt-oss-120b",)


def _prefer(cands: list["ModelInfo"], substrs: tuple[str, ...]) -> Optional[str]:
    for sub in substrs:
        for m in cands:
            if sub in m.id:
                return m.id
    return None


@dataclass(frozen=True)
class ModelInfo:
    id: str
    chat: bool = False
    json_ok: bool = False
    vision: bool = False
    size_hint: float = 0.0   # billions of params inferred from the slug; 0 = unknown


@dataclass(frozen=True)
class ResolvedRoles:
    vlm: Optional[str]
    gen: Optional[str]
    gate1: Optional[str]
    critic: Optional[str]
    gemma_path_used: bool
    notes: tuple[str, ...] = ()

    def as_dict(self) -> dict:
        return {"vlm": self.vlm, "gen": self.gen, "gate1": self.gate1, "critic": self.critic,
                "gemma_path_used": self.gemma_path_used, "notes": list(self.notes)}


def is_gemma(model_id: str) -> bool:
    return "gemma" in model_id.lower()


def size_hint(model_id: str) -> float:
    """Largest 'Nb' token in a slug (gemma-4-26b-a4b -> 26, llama-...-70b -> 70). 0 if none."""
    nums = [float(m) for m in _SIZE_RE.findall(model_id.lower())]
    return max(nums) if nums else 0.0


def resolve_roles(models: list[ModelInfo]) -> ResolvedRoles:
    """Assign each role from reachable capabilities. Pure — the injected probe already ran."""
    chat = [m for m in models if m.chat]
    vision = [m for m in models if m.vision]
    notes: list[str] = []

    def strongest(cands: list[ModelInfo]) -> Optional[str]:
        return max(cands, key=lambda m: m.size_hint).id if cands else None

    # VLM: a Gemma with vision, else a known-good vision fallback, else any vision, else None.
    gemma_vision = [m for m in vision if is_gemma(m.id)]
    if gemma_vision:
        vlm = strongest(gemma_vision)
    elif vision:
        vlm = _prefer(vision, KNOWN_GOOD_VISION) or strongest(vision)
        notes.append(f"vlm: no Gemma vision model; using {vlm}")
    else:
        vlm = None
        notes.append("vlm: no vision model reachable; perception will omit the visual layer")

    # gen: strongest Gemma, else a known-good chat fallback, else strongest chat.
    gemmas = [m for m in chat if is_gemma(m.id)]
    if gemmas:
        gen = strongest(gemmas)
    else:
        gen = _prefer(chat, KNOWN_GOOD_CHAT) or strongest(chat)
        if gen:
            notes.append(f"gen: no Gemma chat model; using {gen}")

    # critic: strongest reachable chat model; prefer a known-good one at equal footing.
    critic = _prefer(chat, KNOWN_GOOD_CHAT) or strongest(chat)

    # gate_1: smallest reachable model that returns parseable JSON; fall back to smallest chat.
    json_models = [m for m in chat if m.json_ok]
    pool = json_models or chat
    if pool:
        gate1 = min(pool, key=lambda m: (m.size_hint if m.size_hint > 0 else 1e9, m.id)).id
        if not json_models:
            notes.append("gate_1: no model confirmed JSON-capable; using smallest chat model")
    else:
        gate1 = None

    gemma_path_used = gen is not None and is_gemma(gen)
    if not gemma_path_used:
        notes.append("gemma_path_used=False: generation is NOT on Gemma for this key")
    return ResolvedRoles(vlm, gen, gate1, critic, gemma_path_used, tuple(notes))


ListFn = Callable[[], Awaitable[list[str]]]
ProbeFn = Callable[[str], Awaitable[tuple[bool, bool, bool]]]  # -> (chat, json_ok, vision)


async def discover(
    list_fn: ListFn, probe_fn: ProbeFn,
    extra_candidates: tuple[str, ...] = KNOWN_GOOD_CANDIDATES,
) -> list[ModelInfo]:
    """List models (plus known-good fallbacks) and probe each. Injected fns keep it offline."""
    ids = list(await list_fn())
    for slug in extra_candidates:
        if slug not in ids:
            ids.append(slug)
    infos: list[ModelInfo] = []
    for mid in ids:
        chat, json_ok, vision = await probe_fn(mid)
        if not (chat or vision):
            continue  # unreachable (e.g. a known-good slug not on this key) — drop it
        infos.append(ModelInfo(id=mid, chat=chat, json_ok=json_ok, vision=vision,
                               size_hint=size_hint(mid)))
    return infos
