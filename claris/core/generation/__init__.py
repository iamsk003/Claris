"""Generation — style-conditioned caption candidates grounded in the ledger.

Public API:
    generate_all(ledger, task, provider, ...)  -> {StyleName: [CaptionCandidate]}
    generate_style(ledger, task, style, shared) -> [CaptionCandidate]

Each style is generated independently (no shared call across styles) and every
candidate that announces a hallucination — cites an evidence ID not in the ledger —
is rejected at parse time before any critic call is spent on it.

Style contracts are declarative YAML under ``styles/`` and are hot-reloadable via
``StyleContractRegistry``.
"""

from claris.core.generation.contracts import (
    FewShot,
    StyleContract,
    StyleContractRegistry,
    load_contract,
    render_system_prompt,
    render_user_prompt,
)
from claris.core.generation.generator import (
    DEFAULT_TEMPERATURES,
    GenerationConfig,
    GenerationOutput,
    generate_all,
    generate_style,
)

__all__ = [
    "FewShot",
    "StyleContract",
    "StyleContractRegistry",
    "load_contract",
    "render_system_prompt",
    "render_user_prompt",
    "DEFAULT_TEMPERATURES",
    "GenerationConfig",
    "GenerationOutput",
    "generate_all",
    "generate_style",
]
