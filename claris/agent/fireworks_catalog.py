"""Real Fireworks catalog probes for capability discovery.

GET /v1/models to list, then 1-token probes per model: a plain text call (chat?), a
JSON-mode call (parseable JSON?), and an image call (vision?). A 404 bills nothing; the
1-token 200s are negligible. This is the only place discovery touches the network; the
resolution logic in ``discovery`` is pure and injected with these at runtime.

The generic vision probe covers kimi-k2p6 automatically: if it accepts image input it
surfaces as vision-capable and becomes an eligible VLM fallback.
"""

from __future__ import annotations

import base64
import json

import httpx

_PNG = base64.b64encode(base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)).decode()


def make_probes(base_url: str, api_key: str, client: httpx.AsyncClient, timeout_s: float = 30.0):
    """Return (list_fn, probe_fn) bound to a Fireworks endpoint."""
    base = base_url.rstrip("/")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    async def list_fn() -> list[str]:
        resp = await client.get(f"{base}/models", headers=headers, timeout=timeout_s)
        resp.raise_for_status()
        data = resp.json().get("data", [])
        return [m.get("id", "") for m in data if m.get("id")]

    async def _post(payload: dict) -> httpx.Response:
        return await client.post(f"{base}/chat/completions", json=payload, headers=headers,
                                 timeout=timeout_s)

    async def probe_fn(model_id: str) -> tuple[bool, bool, bool]:
        text = {"model": model_id, "messages": [{"role": "user", "content": "ok"}], "max_tokens": 1}
        try:
            chat = (await _post(text)).status_code == 200
        except httpx.HTTPError:
            chat = False
        if not chat:
            return False, False, False

        json_ok = False
        try:
            r = await _post({**text, "max_tokens": 20, "messages": [
                {"role": "user", "content": 'Return this JSON exactly: {"ok": 1}'}],
                "response_format": {"type": "json_object"}})
            if r.status_code == 200:
                json.loads(r.json()["choices"][0]["message"]["content"])
                json_ok = True
        except Exception:  # noqa: BLE001 — any failure means "not reliably JSON"
            json_ok = False

        vision = False
        try:
            r = await _post({"model": model_id, "max_tokens": 1, "messages": [{"role": "user",
                "content": [{"type": "text", "text": "ok"},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_PNG}"}}]}]})
            vision = r.status_code == 200
        except httpx.HTTPError:
            vision = False

        return chat, json_ok, vision

    return list_fn, probe_fn
