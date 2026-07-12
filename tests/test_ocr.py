"""The shared OCR engine initializes exactly once, even under concurrent access.

Guards the startup-warm path (`warm_ocr`) and the request path (`get_ocr_engine`, reached via
`run_ocr`) against double initialization when a request lands while the startup task is still
building the engine.
"""

from __future__ import annotations

import threading
import time

import claris.core.perception.ocr as ocrmod
from claris.core.perception.ocr import get_ocr_engine, warm_ocr


def test_engine_built_once_under_concurrent_warm_and_requests(monkeypatch):
    ocrmod._ENGINE = None  # start from a cold process
    builds = {"n": 0}

    def slow_builder(cfg):
        # Simulate PaddleOCR's slow constructor (model download + native init).
        builds["n"] += 1
        time.sleep(0.05)
        return object()  # stand-in engine; identity is what we assert on

    monkeypatch.setattr(ocrmod, "_paddle_engine", slow_builder)

    results: list[object] = []

    def request_worker():
        results.append(get_ocr_engine())

    # One startup warm plus several requests arriving while the download is in flight.
    threads = [threading.Thread(target=warm_ocr)]
    threads += [threading.Thread(target=request_worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    try:
        assert builds["n"] == 1  # exactly one initialization across warm + all requests
        assert len({id(r) for r in results}) == 1  # every request got the same engine
        assert results[0] is ocrmod._ENGINE
    finally:
        ocrmod._ENGINE = None  # reset process-wide global for other tests


def test_second_call_reuses_without_rebuilding(monkeypatch):
    ocrmod._ENGINE = None
    builds = {"n": 0}
    monkeypatch.setattr(ocrmod, "_paddle_engine",
                        lambda cfg: (builds.__setitem__("n", builds["n"] + 1), object())[1])
    try:
        first = get_ocr_engine()
        second = get_ocr_engine()
        warm_ocr()  # startup arriving late must not rebuild either
        assert builds["n"] == 1
        assert first is second is ocrmod._ENGINE
    finally:
        ocrmod._ENGINE = None
