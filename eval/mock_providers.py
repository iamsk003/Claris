"""Deterministic, content-driven mock providers for offline runs.

These are not fixed-score stubs. Each reads the evidence out of the prompt exactly as a
real model would see it (generation embeds the ledger block; the judge and critic are
given the cited evidence), then reasons about grounding with transparent token-overlap
heuristics. This lets the ablation harness run with zero network and still measure a
*real* effect: the gates select more-grounded, better-scored captions than taking the
first sample. The numbers are synthetic (mock generation), but the mechanism is genuine.
"""

from __future__ import annotations

import json
import re
from typing import Optional

from claris.core.providers.base import CompletionResult
from claris.core.schema import ProviderTier, estimate_tokens


def _result(text: str, model: str, prompt: str, system: str, temperature: float,
            seed: int) -> CompletionResult:
    """Build a mock CompletionResult with estimated token counts, so the meter has data."""
    return CompletionResult(
        text=text, model=model, provider_tier=ProviderTier.FIREWORKS_GEMMA,
        seed=seed, temperature=temperature, latency_ms=1.0,
        tokens_in=estimate_tokens(system + prompt), tokens_out=estimate_tokens(text),
    )

_LEDGER_LINE = re.compile(r"\[(E\d+)\]\s+\S+\s+[\d.]+-[\d.]+s\s+\([\d.]+\):\s+(.+)")
_JUDGE_EV = re.compile(r"\[(E\d+)\]\s+\w+:\s+(.+)")
_STYLE_RE = re.compile(r'produce ONLY the "([a-z_]+)" style')
_CAPTION_RE = re.compile(r'Caption(?:\s*\([a-z_]+\))?:\s*"?(.+?)"?\s*$', re.MULTILINE)


def _tier_for(temperature: float) -> int:
    """Quality tier by temperature THRESHOLD, so it is robust to the sample set.

    The lowest-temperature sample (candidate index 0, which the gates-off arm picks) is
    always tier 0 — the ungrounded one — whether the temperatures are cheap [0.4, 0.8] or
    thorough [0.3, 0.6, 0.8, 1.0]. Higher temperatures are progressively richer/grounded.
    """
    if temperature <= 0.5:
        return 0
    if temperature <= 0.7:
        return 1
    if temperature <= 0.9:
        return 2
    return 3
_STOP = {"the", "and", "with", "that", "this", "into", "from", "over", "then", "they",
         "a", "an", "of", "to", "in", "on", "is", "it", "as", "at", "for"}

# Each style gets a distinct opener, suffix, and signature word, so different styles
# produce lexically distinct captions about the same evidence — as real style contracts do.
_STYLE_TEMPLATE = {
    "formal": ("The footage documents", "."),
    "sarcastic": ("Oh wonderful, we simply must admire",
                  ", truly a landmark achievement in doing almost nothing."),
    "humorous_tech": ("Pushing straight to prod:",
                      "; latency nominal, dignity deprecated, rollback unavailable."),
    "humorous_non_tech": ("Bless this whole scene,",
                          ", the sort of thing that makes an ordinary afternoon feel enormous."),
}
_STYLE_SIG = {"formal": "documents", "sarcastic": "wonderful",
              "humorous_tech": "prod", "humorous_non_tech": "bless"}


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 3 and w not in _STOP}


def _overlap(a: str, b: str) -> bool:
    return len(_words(a) & _words(b)) >= 2


def _summarize(content: str) -> str:
    return " ".join(content.rstrip(".").split()[:8]).lower()


def _parse_ledger(prompt: str) -> list[tuple[str, str]]:
    return [(m.group(1), m.group(2).strip()) for m in _LEDGER_LINE.finditer(prompt)]


class MockGenProvider:
    """Generation mock. Tier (from temperature) sets grounding quality.

    tier 0 (temp 0.3) appends an unsupported claim, so it fails gate_1; higher tiers are
    fully grounded and progressively longer/richer, so the critic prefers them. Taking the
    first sample (gates off) therefore lands on the ungrounded tier-0 caption.
    """

    tier = ProviderTier.FIREWORKS_GEMMA
    model = "mock-gemma"

    async def complete(self, *, system, prompt, temperature, seed, model=None,
                       max_tokens=512, timeout_s=90.0, json_mode=True) -> CompletionResult:
        style_m = _STYLE_RE.search(system)
        style = style_m.group(1) if style_m else "formal"
        tier = _tier_for(temperature)
        evidence = _parse_ledger(prompt)

        chosen = evidence[: tier + 1] if evidence else []
        cited = [eid for eid, _ in chosen]
        body = "; ".join(_summarize(c) for _, c in chosen) or "something occurs"
        opener, suffix = _STYLE_TEMPLATE.get(style, ("A clip of", "."))
        text = f"{opener} {body}{suffix}"
        if tier == 0:
            text += " A dragon circles overhead."  # unsupported claim -> gate_1 catches it
        payload = json.dumps({"caption": text, "cited_evidence_ids": cited})
        return _result(payload, self.model, prompt, system, temperature, seed)


class MockJudge:
    """gate_1 judge mock. Splits the caption into sentences and checks token overlap
    against the cited evidence; a sentence with no overlap is 'not_supported'."""

    tier = ProviderTier.FIREWORKS_GEMMA
    model = "mock-judge"

    async def complete(self, *, system, prompt, temperature, seed, model=None,
                       max_tokens=512, timeout_s=90.0, json_mode=True) -> CompletionResult:
        cap_m = _CAPTION_RE.search(prompt)
        caption = cap_m.group(1) if cap_m else ""
        evidence = [(m.group(1), m.group(2).strip()) for m in _JUDGE_EV.finditer(prompt)]
        assessments = []
        for sentence in [s.strip() for s in re.split(r"(?<=[.!?])\s+", caption) if s.strip()]:
            supporting = [eid for eid, content in evidence if _overlap(sentence, content)]
            verdict = "entailed" if supporting else "not_supported"
            assessments.append({"claim": sentence, "verdict": verdict, "supporting_ids": supporting})
        return _result(json.dumps(assessments), self.model, prompt, system, temperature, seed)


class MockCritic:
    """gate_2 critic mock. Accuracy from grounding overlap, tone from style-word presence,
    naturalness from length; deterministic per (caption, ledger)."""

    tier = ProviderTier.FIREWORKS_GEMMA
    model = "mock-critic"

    async def complete(self, *, system, prompt, temperature, seed, model=None,
                       max_tokens=512, timeout_s=90.0, json_mode=True) -> CompletionResult:
        style_m = re.search(r"style:\s*([a-z_]+)", system)
        style = style_m.group(1) if style_m else "formal"
        cap_m = _CAPTION_RE.search(prompt)
        caption = cap_m.group(1) if cap_m else ""
        evidence = _parse_ledger(prompt)
        ev_blob = " ".join(c for _, c in evidence)

        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", caption) if s.strip()]
        grounded = sum(1 for s in sentences if _overlap(s, ev_blob))
        acc_frac = grounded / len(sentences) if sentences else 0.0
        accuracy = 1.0 + 4.0 * acc_frac
        tone = 4.0 + (0.5 if _STYLE_SIG.get(style, "") in caption.lower() else -0.5)
        distinct = 3.5
        naturalness = 2.0 + min(3.0, len(caption.split()) / 8.0)
        obj = {
            "accuracy": round(accuracy, 2), "tone_fidelity": round(tone, 2),
            "style_distinctness": distinct, "naturalness": round(naturalness, 2),
            "accuracy_reason": f"{grounded}/{len(sentences)} sentences grounded in evidence",
            "tone_reason": f"graded against the {style} contract",
            "distinctness_reason": "reads as its style", "naturalness_reason": "fluent",
        }
        return _result(json.dumps(obj), self.model, prompt, system, temperature, seed)


class MockHeldOutJudge:
    """Independent measurement judge for offline runs (see eval/judge.py).

    Deliberately shares NO scoring signal with MockCritic: it grades accuracy purely by
    grounding overlap against the ledger (objective), and tone by a flat contract-based
    check — it does not know or reward the critic's style-signature trick. So the ablation
    delta it reports comes from grounding, not from grading the paper with its own pen.
    """

    tier = ProviderTier.FIREWORKS_GEMMA
    model = "mock-held-out-judge"

    async def complete(self, *, system, prompt, temperature, seed, model=None,
                       max_tokens=512, timeout_s=90.0, json_mode=True) -> CompletionResult:
        cap_m = re.search(r'CAPTION TO GRADE:\s*"(.+?)"', prompt, re.DOTALL)
        caption = cap_m.group(1) if cap_m else ""
        evidence = _parse_ledger(prompt)
        ev_blob = " ".join(c for _, c in evidence)
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", caption) if s.strip()]
        grounded = sum(1 for s in sentences if _overlap(s, ev_blob))
        acc_frac = grounded / len(sentences) if sentences else 0.0
        accuracy = 1.0 + 4.0 * acc_frac
        tone = 4.0 - (0.5 if len(caption.split()) < 5 else 0.0)  # flat, style-agnostic
        obj = {
            "accuracy": round(accuracy, 2), "tone_fidelity": round(tone, 2),
            "accuracy_reason": f"{grounded}/{len(sentences)} sentences supported by evidence",
            "tone_reason": "style-agnostic fluency check",
        }
        return _result(json.dumps(obj), self.model, prompt, system, temperature, seed)


class MockVisionProvider:
    """Vision mock. Returns a valid frame-description array so preflight + parsing pass."""

    tier = ProviderTier.FIREWORKS_GEMMA
    model = "mock-gemma-vlm"

    async def complete(self, *, system, prompt, images, temperature, seed, model=None,
                       max_tokens=1024, timeout_s=90.0, json_mode=True) -> CompletionResult:
        n = max(1, len(images))
        frames = [{"setting": "a scene", "subjects": [], "actions": [], "objects": [],
                   "mood": "", "notable_detail": ""} for _ in range(n)]
        return _result(json.dumps(frames), self.model, prompt, system, temperature, seed)


def mock_embed(texts: list[str]) -> list[list[float]]:
    """Bag-of-words embedding over the batch vocabulary. Deterministic, no model."""
    vocab: dict[str, int] = {}
    tokenized = []
    for t in texts:
        toks = _words(t)
        tokenized.append(toks)
        for w in toks:
            vocab.setdefault(w, len(vocab))
    dim = max(1, len(vocab))
    vectors = []
    for toks in tokenized:
        v = [0.0] * dim
        for w in toks:
            v[vocab[w]] = 1.0
        vectors.append(v)
    return vectors


def mock_providers():
    """A Providers bundle wired to the deterministic mocks."""
    from claris.core.pipeline import Providers

    return Providers(
        gen_provider=MockGenProvider(), judge=MockJudge(), critic=MockCritic(),
        vision_provider=MockVisionProvider(), embed_fn=mock_embed,
    )
