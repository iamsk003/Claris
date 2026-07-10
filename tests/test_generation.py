"""Generation tests with an ADVERSARIAL fake provider.

The prior suite was content-blind: the mock cited ``next(iter(ledger.evidence_ids))``
and asserted a subset, so it passed for any non-empty ledger and never exercised the
hallucination gate against a mock capable of lying. This suite makes the provider
hostile and pins down exactly what the *parse-time* gate catches versus what only a
future semantic (entailment) gate can catch.

Two classes of adversarial output:

  PARSE-TIME REJECT — the existing gate must drop these (0 candidates):
    2. cites an ID not in the ledger (ev_0099)
    3. cites nothing at all (empty cited_evidence_ids)
    5. valid JSON but empty caption

  SEMANTIC FAILURE — these SURVIVE parse; catching them is Prompt 3's entailment gate.
  The survival assertions below are the executable specification for that gate:
    1. cites a real ID whose content contradicts the caption
    4. correct citation, but the caption asserts an object in no evidence item
    6. cites every ID in the ledger indiscriminately

Nothing in claris/core/generation/ is modified. Where a case reveals that the current
gate does not do what it should, the test is marked xfail(strict) and reported.
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path

import pytest

from claris.core.generation import (
    GenerationConfig,
    StyleContractRegistry,
    generate_all,
)
from claris.core.generation.contracts import render_system_prompt
from claris.core.observability import ListSink
from claris.core.schema import (
    ALL_STYLES,
    CaptionCandidate,
    EvidenceItem,
    EvidenceLedger,
    StyleName,
    Task,
    VideoMeta,
)
from eval.harness import load_golden_ledgers
from tests.fakes import FakeProvider, valid_json

ONE_STYLE = (StyleName.FORMAL,)


def _cfg(n: int = 1) -> GenerationConfig:
    # No pacing, generous concurrency: keep the offline suite fast.
    return GenerationConfig(n=n, rate_per_sec=0.0, max_concurrency=16)


def _known_ledger() -> EvidenceLedger:
    """A small ledger with known content, so contradictions can be crafted precisely."""
    meta = VideoMeta(video_sha256="c" * 64, duration_s=30.0, has_audio=True)
    items = (
        EvidenceItem(
            id="E001", kind="visual", t_start=0.0, t_end=5.0,
            content="A chef sears a steak in a cast-iron skillet.",
            confidence=0.9, source_model="gemma-3-vlm",
        ),
        EvidenceItem(
            id="E002", kind="speech", t_start=6.0, t_end=9.0,
            content="Let it rest for about five minutes.",
            confidence=0.88, source_model="faster-whisper",
        ),
        EvidenceItem(
            id="E003", kind="audio_event", t_start=0.0, t_end=10.0,
            content="Loud sizzling sound.",
            confidence=0.7, source_model="librosa-heuristic",
        ),
    )
    return EvidenceLedger(
        ledger_id="led_known", task_id="known", video_sha256="c" * 64,
        video_meta=meta, items=items,
    )


def _run(ledger: EvidenceLedger, text: str, n: int = 1) -> list[CaptionCandidate]:
    """Run generation for a single style against a provider that always returns ``text``."""
    task = Task(task_id=ledger.task_id, video_path="x.mp4", styles=ONE_STYLE)
    provider = FakeProvider(lambda *_: text)
    out = asyncio.run(
        generate_all(ledger, task, provider, config=_cfg(n), sink=ListSink(), run_id="adv")
    )
    return out[StyleName.FORMAL]


# ----------------------------------------------------------------- parse-time REJECT


def test_case2_unknown_id_is_rejected():
    cands = _run(_known_ledger(), valid_json("A chef prepares a dish.", ["ev_0099"]))
    assert cands == []  # ev_0099 is not in the ledger -> announced hallucination


def test_case5_empty_caption_is_rejected():
    cands = _run(_known_ledger(), valid_json("", ["E001"]))
    assert cands == []  # empty caption -> parse reject


def test_case3_empty_citations_is_rejected():
    # A caption grounded in zero evidence is rejected at parse time.
    cands = _run(_known_ledger(), valid_json("A chef cooks a meal.", []))
    assert cands == []


# ------------------------------------------------------------- SEMANTIC (survive parse)
# These assertions ARE the spec for Prompt 3's entailment gate. They must survive the
# parse-time gate today; a later semantic gate is what should reject them.


def test_case1_contradiction_survives_parse():
    # Cites E001 ("sears a steak") but describes poaching a fish: a real citation whose
    # content contradicts the caption. Parse gate cannot see this; entailment gate must.
    cands = _run(
        _known_ledger(),
        valid_json("The chef gently poaches a whole fish in simmering broth.", ["E001"]),
    )
    assert len(cands) == 1
    assert cands[0].evidence_ids == ("E001",)


def test_case4_unsupported_object_with_real_citation_survives_parse():
    # 'golden retriever' is in no evidence item; the citation E001 is real. This is the
    # realistic hallucination: correct citation, unsupported claim.
    cands = _run(
        _known_ledger(),
        valid_json(
            "A chef sears a steak while a golden retriever watches from the counter.",
            ["E001"],
        ),
    )
    assert len(cands) == 1
    assert "retriever" in cands[0].text


def test_case6_indiscriminate_citation_survives_parse():
    # Cites every ID regardless of relevance. All IDs are real, so parse passes; the
    # entailment gate must penalize citing evidence the caption does not actually use.
    cands = _run(
        _known_ledger(),
        valid_json("Some things occur in this video.", ["E001", "E002", "E003"]),
    )
    assert len(cands) == 1
    assert set(cands[0].evidence_ids) == {"E001", "E002", "E003"}


# ----------------------------------------------------------------- structural / parsing


def test_malformed_json_is_rejected():
    assert _run(_known_ledger(), "not json at all, sorry") == []


def test_json_wrapped_in_code_fence_is_recovered():
    fenced = '```json\n{"caption": "Fenced but fine.", "cited_evidence_ids": ["E001"]}\n```'
    cands = _run(_known_ledger(), fenced, n=4)
    assert len(cands) == 4


def test_to_prompt_block_contains_every_evidence_id():
    # Generation embeds to_prompt_block() verbatim and relies on every ID being visible
    # so the model can cite it. Nothing else enforces this; pin it here.
    ledgers = load_golden_ledgers() + [_known_ledger()]
    for ledger in ledgers:
        block = ledger.to_prompt_block()
        for it in ledger.items:
            assert it.id in block, f"{it.id} missing from to_prompt_block of {ledger.ledger_id}"


# ----------------------------------------------------------------- contracts / wiring


def test_all_contracts_load_and_render():
    reg = StyleContractRegistry()
    assert set(reg.styles()) == set(ALL_STYLES)
    for style in ALL_STYLES:
        contract = reg.get(style)
        assert contract.name == style
        prompt = render_system_prompt(contract)
        assert "STRICT JSON" in prompt and "cited_evidence_ids" in prompt
    non_tech = reg.get(ALL_STYLES[3])
    assert any("technical" in fm.lower() for fm in non_tech.forbidden_moves)
    assert reg.get(ALL_STYLES[0]).humor_device is None


def test_cooperative_smoke_four_styles_four_candidates():
    # A deliberately cooperative wiring check: every style produces 4 grounded candidates
    # with distinct seeds. Kept as a smoke test only; the adversarial cases above are the
    # ones that actually exercise the gate.
    ledger = load_golden_ledgers()[0]
    task = Task(task_id=ledger.task_id, video_path="x.mp4", styles=ALL_STYLES)
    real_id = sorted(ledger.evidence_ids)[0]
    provider = FakeProvider(lambda *_: valid_json("A grounded caption.", [real_id]))
    out = asyncio.run(generate_all(ledger, task, provider, config=_cfg(4), run_id="smoke"))
    assert set(out) == set(ALL_STYLES)
    for style in ALL_STYLES:
        assert len(out[style]) == 4
        assert all(set(c.evidence_ids) <= ledger.evidence_ids for c in out[style])


def test_contracts_are_hot_reloadable(tmp_path: Path):
    src = Path("claris/core/generation/styles/formal.yaml")
    dst = tmp_path / "formal.yaml"
    dst.write_text(src.read_text())

    reg = StyleContractRegistry(styles_dir=tmp_path)
    before = reg.get(StyleName.FORMAL)
    assert "hot-reload marker" not in before.intent

    dst.write_text(dst.read_text().replace("intent: >", "intent: >\n  hot-reload marker.", 1))
    future = time.time() + 5
    os.utime(dst, (future, future))

    after = reg.get(StyleName.FORMAL)
    assert "hot-reload marker" in after.intent


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
