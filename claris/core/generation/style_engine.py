"""Deprecated shim.

The real style-conditioned generation lives in ``generator.py`` (``generate_style`` /
``generate_all``) and the declarative style definitions live in ``contracts.py`` +
``styles/*.yaml``. This module is kept only so older imports do not break.
"""

from __future__ import annotations

from claris.core.generation.generator import generate_all, generate_style

__all__ = ["generate_style", "generate_all"]
