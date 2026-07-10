"""Providers — the ONLY place network inference happens.

Fireworks AI API (default), optional local vLLM backend. Every call is seeded,
temperature-explicit, retried with backoff+jitter, and logged as a RunEvent.

Fallbacks below the preferred model set ``degraded=True``.
"""
