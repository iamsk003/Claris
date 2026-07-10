"""Style contracts — the declarative definition of each caption style.

A style is a contract, not a paragraph of prose. Each ``styles/*.yaml`` file is
parsed into a frozen ``StyleContract`` and rendered into a system prompt. Contracts
are **hot-reloadable**: ``StyleContractRegistry.get()`` re-reads a file whenever its
mtime changes, so the four YAML files can be edited without touching code.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from claris.core.schema import ALL_STYLES, StyleName

DEFAULT_STYLES_DIR = Path(__file__).parent / "styles"

# Matches ledger IDs like [ev_0003] inside a few-shot excerpt.
_ID_RE = re.compile(r"\[(ev_[0-9A-Za-z_]+)\]")


class FewShot(BaseModel):
    """One worked example: a ledger excerpt and the caption it should yield."""

    model_config = ConfigDict(frozen=True)

    ledger_excerpt: str = Field(..., min_length=1)
    caption: str = Field(..., min_length=1)

    def cited_ids(self) -> list[str]:
        """IDs referenced in the excerpt, in first-seen order.

        Used to render the demonstrated JSON output so the few-shot teaches both the
        wording *and* the citation format.
        """
        seen: list[str] = []
        for m in _ID_RE.findall(self.ledger_excerpt):
            if m not in seen:
                seen.append(m)
        return seen


class StyleContract(BaseModel):
    """The complete, declarative definition of one caption style."""

    model_config = ConfigDict(frozen=True)

    name: StyleName
    intent: str = Field(..., min_length=1)
    voice: str = Field(..., min_length=1)
    sentence_length_target: str = Field(..., min_length=1)
    lexicon_prefer: tuple[str, ...] = Field(default_factory=tuple)
    lexicon_forbid: tuple[str, ...] = Field(default_factory=tuple)
    humor_device: Optional[str] = None
    forbidden_moves: tuple[str, ...] = Field(default_factory=tuple)
    length_words: tuple[int, int]
    few_shot: tuple[FewShot, ...]

    @field_validator("length_words")
    @classmethod
    def _length_ordered(cls, v: tuple[int, int]) -> tuple[int, int]:
        lo, hi = v
        if lo <= 0 or hi <= 0 or hi < lo:
            raise ValueError(f"length_words must be positive and ordered, got {v}")
        return v

    @field_validator("few_shot")
    @classmethod
    def _needs_examples(cls, v: tuple[FewShot, ...]) -> tuple[FewShot, ...]:
        if len(v) < 1:
            raise ValueError("a style contract needs at least one few-shot example")
        return v


def load_contract(path: Path) -> StyleContract:
    """Parse and validate a single style YAML file."""
    data = yaml.safe_load(path.read_text())
    return StyleContract.model_validate(data)


class StyleContractRegistry:
    """Loads style contracts from a directory and hot-reloads on file change.

    ``get(style)`` stats the backing file and reparses it only when its mtime has
    advanced, so edits to the YAML take effect on the next generation with no code
    change and no restart.
    """

    def __init__(self, styles_dir: Optional[Path] = None) -> None:
        self.styles_dir = Path(styles_dir) if styles_dir else DEFAULT_STYLES_DIR
        self._paths: dict[StyleName, Path] = {}
        self._cache: dict[StyleName, tuple[float, StyleContract]] = {}
        self._discover()

    def _discover(self) -> None:
        self._paths = {}
        for p in sorted(self.styles_dir.glob("*.yaml")):
            data = yaml.safe_load(p.read_text())
            try:
                name = StyleName(data["name"])
            except (KeyError, ValueError) as exc:
                raise ValueError(f"{p} has a missing or invalid 'name'") from exc
            self._paths[name] = p

    def get(self, style: StyleName) -> StyleContract:
        path = self._paths.get(style)
        if path is None:
            # A file may have appeared since construction; rediscover once.
            self._discover()
            path = self._paths.get(style)
        if path is None:
            raise KeyError(f"No style contract found for {style.value} in {self.styles_dir}")

        mtime = path.stat().st_mtime
        cached = self._cache.get(style)
        if cached is not None and cached[0] == mtime:
            return cached[1]

        contract = load_contract(path)
        if contract.name != style:
            raise ValueError(
                f"{path} declares name={contract.name.value} but is loaded as {style.value}"
            )
        self._cache[style] = (mtime, contract)
        return contract

    def styles(self) -> tuple[StyleName, ...]:
        return tuple(s for s in ALL_STYLES if s in self._paths)

    def all(self) -> dict[StyleName, StyleContract]:
        return {s: self.get(s) for s in self.styles()}


# --------------------------------------------------------------------------- #
# Prompt rendering
# --------------------------------------------------------------------------- #


def _bullet(items: tuple[str, ...]) -> str:
    return "\n".join(f"  - {it}" for it in items) if items else "  (none)"


def render_system_prompt(contract: StyleContract) -> str:
    """Render a style contract into a strict system prompt.

    The prompt states the contract, mandates the JSON output shape, forbids inventing
    evidence IDs, and shows the few-shot examples as ledger -> JSON pairs.
    """
    parts: list[str] = []
    parts.append(
        f'You are a video caption writer. You produce ONLY the "{contract.name.value}" '
        f"style. You do not write in any other style."
    )
    parts.append(f"INTENT:\n  {contract.intent.strip()}")
    parts.append(f"VOICE:\n  {contract.voice.strip()}")
    parts.append(f"SENTENCE LENGTH:\n  {contract.sentence_length_target.strip()}")
    lo, hi = contract.length_words
    parts.append(f"TOTAL LENGTH:\n  Between {lo} and {hi} words.")
    if contract.humor_device:
        parts.append(f"HUMOR DEVICE:\n  {contract.humor_device.strip()}")
    else:
        parts.append("HUMOR DEVICE:\n  None. Do not attempt to be funny.")
    parts.append("PREFER words and moves like:\n" + _bullet(contract.lexicon_prefer))
    parts.append("NEVER use these words:\n" + _bullet(contract.lexicon_forbid))
    parts.append("FORBIDDEN MOVES (these disqualify the caption):\n" + _bullet(contract.forbidden_moves))

    parts.append(
        "GROUNDING RULES:\n"
        "  - You receive a VIDEO EVIDENCE LEDGER. Each fact has an ID like [ev_0003].\n"
        "  - Every claim in your caption must be supported by at least one ledger item.\n"
        "  - Do not state anything the ledger does not support. If unsure, leave it out.\n"
        '  - Output STRICT JSON and NOTHING else: '
        '{"caption": "<text>", "cited_evidence_ids": ["ev_0003", ...]}\n'
        "  - cited_evidence_ids MUST be a subset of the IDs in the ledger. Never invent an ID."
    )

    examples: list[str] = []
    for fs in contract.few_shot:
        demo = json.dumps(
            {"caption": " ".join(fs.caption.split()), "cited_evidence_ids": fs.cited_ids()},
            ensure_ascii=False,
        )
        examples.append(
            "Ledger:\n" + fs.ledger_excerpt.strip() + "\nOutput:\n" + demo
        )
    parts.append("EXAMPLES:\n\n" + "\n\n".join(examples))

    return "\n\n".join(parts)


def render_user_prompt(ledger_block: str, task_note: Optional[str] = None) -> str:
    """Render the per-clip user message from the ledger block and optional note."""
    note = f"\n\nADDITIONAL CONTEXT:\n{task_note.strip()}" if task_note else ""
    return (
        f"{ledger_block}{note}\n\n"
        "Write exactly one caption for this video in your assigned style. "
        "Respond with the JSON object only."
    )
