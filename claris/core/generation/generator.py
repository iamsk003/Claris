"""Style-conditioned caption generation with announced-hallucination rejection.

For each style, INDEPENDENTLY (never one call for all four — cross-contamination
between styles is real), we:

  1. build the prompt: system = the style contract, user = the ledger block + task;
  2. require the model to emit JSON ``{"caption": str, "cited_evidence_ids": [str]}``;
  3. sample N candidates at distinct temperatures and distinct seeds;
  4. reject at PARSE TIME any candidate that cites an ID absent from the ledger — the
     model announced its own hallucination, so we spend no critic call on it.

Concurrency: a shared ``asyncio.Semaphore`` bounds in-flight calls and a global
``AsyncRateLimiter`` paces them, so 4 styles x 4 candidates = 16 calls do not stampede
the endpoint. Every call and every rejection is logged as a structured ``RunEvent``.

The provider is injected (``ChatProvider``), so this module is fully testable offline
with a fake and knows nothing about Fireworks specifically.
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import Optional

from pydantic import BaseModel, ValidationError

from claris.core.generation.contracts import (
    StyleContractRegistry,
    render_system_prompt,
    render_user_prompt,
)
from claris.core.observability import EventSink, NullSink
from claris.core.providers.base import AsyncRateLimiter, ChatProvider
from claris.core.schema import (
    ALL_STYLES,
    CaptionCandidate,
    EvidenceLedger,
    RunEvent,
    StyleName,
    Task,
    sha256_str,
    utcnow,
)

# Cheap mode (the development default): 2 samples. Thorough mode: 4 samples, used once
# for the final table. The critic later picks the best; here we just spread the samples.
CHEAP_TEMPERATURES: tuple[float, ...] = (0.4, 0.8)
THOROUGH_TEMPERATURES: tuple[float, ...] = (0.3, 0.6, 0.8, 1.0)
DEFAULT_TEMPERATURES: tuple[float, ...] = CHEAP_TEMPERATURES

_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


class GenerationOutput(BaseModel):
    """The exact JSON shape the model is required to emit."""

    caption: str
    cited_evidence_ids: list[str] = []


@dataclass(frozen=True)
class GenerationConfig:
    """Knobs for a generation run. Defaults are CHEAP mode (2 samples)."""

    n: int = 2
    temperatures: tuple[float, ...] = CHEAP_TEMPERATURES
    base_seed: int = 1337
    max_concurrency: int = 8
    rate_per_sec: float = 4.0
    # Headroom so a reasoning model (e.g. gpt-oss on a fallback key, which rejects
    # reasoning_effort=none) can emit its thinking AND the JSON caption within budget
    # rather than running out before the answer and falling back to a template.
    max_tokens: int = 2048
    timeout_s: float = 90.0

    # Sequential mode: generate styles one at a time, each seeing what the previous ones
    # wrote, to PREVENT tone collisions rather than detect them after the fact (gate_3).
    # Order is deliberate: formal fixes the facts first; sarcastic and humorous_non_tech
    # are the colliding pair and are kept maximally apart.
    sequential: bool = False
    sequential_order: tuple[StyleName, ...] = (
        StyleName.FORMAL, StyleName.HUMOROUS_TECH, StyleName.SARCASTIC, StyleName.HUMOROUS_NON_TECH,
    )

    @classmethod
    def thorough(cls) -> "GenerationConfig":
        """N=4 at four temperatures. Use once, for the final table."""
        return cls(n=4, temperatures=THOROUGH_TEMPERATURES)

    def temperature_for(self, i: int) -> float:
        return self.temperatures[i % len(self.temperatures)]

    def seed_for(self, style: StyleName, i: int) -> int:
        # Deterministic and distinct per (style, candidate index) for replayability.
        style_offset = ALL_STYLES.index(style) if style in ALL_STYLES else 0
        return self.base_seed + style_offset * 1000 + i


@dataclass
class _Shared:
    """Concurrency + logging context shared across every call in one run."""

    provider: ChatProvider
    registry: StyleContractRegistry
    cfg: GenerationConfig
    sink: EventSink
    run_id: str
    semaphore: asyncio.Semaphore
    limiter: AsyncRateLimiter
    task_id: Optional[str] = None


def _extract_json(text: str) -> Optional[GenerationOutput]:
    """Parse the model output into a GenerationOutput, tolerating minor wrapping.

    Tries a direct parse first, then strips ```json fences and finally falls back to
    the first ``{...}`` span. Returns None if nothing valid can be recovered.
    """
    candidates: list[str] = []
    stripped = text.strip()
    candidates.append(stripped)
    if stripped.startswith("```"):
        inner = stripped.strip("`")
        inner = re.sub(r"^json\s*", "", inner, flags=re.IGNORECASE)
        candidates.append(inner.strip())
    m = _JSON_OBJ_RE.search(stripped)
    if m:
        candidates.append(m.group(0))

    for cand in candidates:
        try:
            data = json.loads(cand)
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(data, dict):
            continue
        try:
            return GenerationOutput.model_validate(data)
        except ValidationError:
            continue
    return None


async def _one_candidate(
    shared: _Shared,
    style: StyleName,
    system: str,
    user: str,
    ledger: EvidenceLedger,
    index: int,
) -> Optional[CaptionCandidate]:
    """Generate, parse, and gate a single candidate. Returns None if rejected."""
    cfg = shared.cfg
    temperature = cfg.temperature_for(index)
    seed = cfg.seed_for(style, index)
    prompt_hash = sha256_str(system + "\x00" + user)

    async with shared.semaphore:
        await shared.limiter.acquire()
        try:
            result = await asyncio.wait_for(
                shared.provider.complete(
                    system=system,
                    prompt=user,
                    temperature=temperature,
                    seed=seed,
                    max_tokens=cfg.max_tokens,
                    timeout_s=cfg.timeout_s,
                    json_mode=True,
                ),
                timeout=cfg.timeout_s + 5.0,
            )
        except Exception as exc:  # noqa: BLE001 — one bad call must not kill the batch
            shared.sink.emit(
                RunEvent(
                    run_id=shared.run_id,
                    event_id=f"{shared.run_id}:{style.value}:{index}:error",
                    stage="generation",
                    event_type="llm_call_failed",
                    level="error",  # type: ignore[arg-type]
                    task_id=shared.task_id,
                    model=shared.provider.model,
                    provider_tier=shared.provider.tier,
                    prompt_hash=prompt_hash,
                    seed=seed,
                    temperature=temperature,
                    payload={"style": style.value, "error": repr(exc)},
                )
            )
            return None

    shared.sink.emit(
        RunEvent(
            run_id=shared.run_id,
            event_id=f"{shared.run_id}:{style.value}:{index}:call",
            stage="generation",
            event_type="llm_call",
            task_id=shared.task_id,
            model=result.model,
            provider_tier=result.provider_tier,
            prompt_hash=prompt_hash,
            seed=seed,
            temperature=temperature,
            latency_ms=result.latency_ms,
            tokens_in=result.tokens_in,
            tokens_out=result.tokens_out,
            payload={"style": style.value, "candidate_index": index, "cost_stage": "generation"},
        )
    )

    parsed = _extract_json(result.text)
    # A caption grounded in zero evidence is exactly the failure this gate exists to
    # catch, so an empty cited_evidence_ids is a parse-time reject alongside an
    # unparseable response or an empty caption.
    if parsed is None or not parsed.caption.strip() or not parsed.cited_evidence_ids:
        shared.sink.emit(
            RunEvent(
                run_id=shared.run_id,
                event_id=f"{shared.run_id}:{style.value}:{index}:reject_parse",
                stage="generation",
                event_type="candidate_rejected",
                level="warn",  # type: ignore[arg-type]
                task_id=shared.task_id,
                model=result.model,
                seed=seed,
                temperature=temperature,
                payload={"style": style.value, "reason": "unparseable_empty_or_uncited"},
            )
        )
        return None

    # Announced hallucination: any cited ID not in the ledger. Reject before spending
    # a critic call on it.
    unknown = [cid for cid in parsed.cited_evidence_ids if not ledger.has(cid)]
    if unknown:
        shared.sink.emit(
            RunEvent(
                run_id=shared.run_id,
                event_id=f"{shared.run_id}:{style.value}:{index}:reject_halluc",
                stage="generation",
                event_type="candidate_rejected",
                level="warn",  # type: ignore[arg-type]
                task_id=shared.task_id,
                model=result.model,
                seed=seed,
                temperature=temperature,
                payload={
                    "style": style.value,
                    "reason": "cited_unknown_evidence",
                    "unknown_ids": unknown,
                },
            )
        )
        return None

    return CaptionCandidate(
        candidate_id=f"cand_{style.value}_{seed}",
        style=style,
        text=parsed.caption.strip(),
        evidence_ids=tuple(dict.fromkeys(parsed.cited_evidence_ids)),
        temperature=temperature,
        seed=seed,
        model=result.model,
        provider_tier=result.provider_tier,
    )


async def generate_style(
    ledger: EvidenceLedger,
    task: Task,
    style: StyleName,
    shared: _Shared,
) -> list[CaptionCandidate]:
    """Produce up to ``cfg.n`` grounded candidates for a single style.

    The returned list contains only candidates that parsed and passed the
    announced-hallucination gate; it may be shorter than ``n`` (or empty).
    """
    contract = shared.registry.get(style)  # re-read from disk if the YAML changed
    system = render_system_prompt(contract)
    user = render_user_prompt(ledger.to_prompt_block(), task.notes)

    tasks = [
        _one_candidate(shared, style, system, user, ledger, i)
        for i in range(shared.cfg.n)
    ]
    results = await asyncio.gather(*tasks)
    return [c for c in results if c is not None]


async def generate_all(
    ledger: EvidenceLedger,
    task: Task,
    provider: ChatProvider,
    *,
    registry: Optional[StyleContractRegistry] = None,
    config: Optional[GenerationConfig] = None,
    sink: Optional[EventSink] = None,
    run_id: Optional[str] = None,
    styles: Optional[tuple[StyleName, ...]] = None,
) -> dict[StyleName, list[CaptionCandidate]]:
    """Generate candidates for every requested style, concurrently but bounded.

    Each style is generated with its own prompt (no shared call), all sharing one
    semaphore and one rate limiter so the whole clip's 16 calls are paced together.
    Returns ``{style: [candidates]}``; a style with no surviving candidate maps to [].
    """
    cfg = config or GenerationConfig()
    registry = registry or StyleContractRegistry()
    sink = sink or NullSink()
    run_id = run_id or f"gen_{task.task_id}_{utcnow().strftime('%Y%m%dT%H%M%S')}"
    styles = styles or task.styles

    shared = _Shared(
        provider=provider,
        registry=registry,
        cfg=cfg,
        sink=sink,
        run_id=run_id,
        semaphore=asyncio.Semaphore(cfg.max_concurrency),
        limiter=AsyncRateLimiter(cfg.rate_per_sec),
        task_id=task.task_id,
    )

    per_style = await asyncio.gather(
        *(generate_style(ledger, task, style, shared) for style in styles)
    )
    return {style: cands for style, cands in zip(styles, per_style)}


__all__ = [
    "GenerationConfig",
    "GenerationOutput",
    "generate_all",
    "generate_style",
    "DEFAULT_TEMPERATURES",
]
