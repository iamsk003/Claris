"""FastAPI surface over the CLARIS captioning engine.

This is the *view layer* only. It does not caption anything itself: it stores an uploaded
clip, hands it to the existing pipeline, streams progress, and returns the result.

Reuse map (nothing here reimplements the engine):
  * providers/model discovery -> ``claris.agent.main.resolve_providers``
  * perception (video -> EvidenceLedger) -> ``claris.core.perception.build_ledger``
  * generation + verification (ledger -> TaskResult) -> ``claris.core.pipeline.run_from_ledger``

``build_ledger`` + ``run_from_ledger`` are the exact two calls ``run_pipeline`` composes; we
call them directly so the intermediate ``EvidenceLedger`` can be returned to the frontend
(``run_pipeline`` returns only the ``TaskResult``). No stage logic is duplicated.

The frontend contract (a clip is uploaded, then a run is started on it):
  POST /api/clips                 -> {"clip_id"}     upload only, no processing yet
  POST /api/clips/{clip_id}/run   -> {"run_id"}      start the pipeline on that clip
  GET  /api/runs/{run_id}         -> RunResult envelope (once finished)
  WS   /api/runs/{run_id}/events  -> progress events: Upload, Extract Frames, Speech
                                     Recognition, OCR, Scene Understanding,
                                     Caption Generation, Finished
  GET  /api/clips/{clip_id}/video -> the stored clip (served while the clip exists)
  GET  /health                    -> {"status": "ok"}

Compatibility aliases (same handlers): POST /upload, GET /result/{run_id}, WS /ws/{run_id}.

Design notes:
  * Heavy engine imports (torch / whisper / opencv) are lazy — done inside the worker, not at
    module import — so the app boots instantly and /health answers before any model loads.
  * The blocking pipeline runs in a worker thread with its own event loop; progress is
    bridged back to the server loop with ``call_soon_threadsafe``. The server loop is never
    blocked, so the WebSocket and health checks stay responsive during a long run.
  * No module-level mutable state: clips and runs live on ``app.state`` in bounded
    registries; a clip's temp file is removed when the clip is evicted or on shutdown.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

# --------------------------------------------------------------------------- #
# Structured logging
# --------------------------------------------------------------------------- #

_LOG = logging.getLogger("claris.api")


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key, value in getattr(record, "extra_fields", {}).items():
            payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def _log(level: int, msg: str, **fields: Any) -> None:
    _LOG.log(level, msg, extra={"extra_fields": fields})


# --------------------------------------------------------------------------- #
# Progress vocabulary — the seven stages the frontend renders
# --------------------------------------------------------------------------- #

STAGE_UPLOAD = "Upload"
STAGE_FRAMES = "Extract Frames"
STAGE_SPEECH = "Speech Recognition"
STAGE_OCR = "OCR"
STAGE_SCENE = "Scene Understanding"
STAGE_CAPTION = "Caption Generation"
STAGE_FINISHED = "Finished"
# The terminal event the frontend keys completion on (Processing.tsx routes to results when
# it receives ``stage === "done"``). Every run — success, degraded, error, or timeout — must
# end by emitting this, which also flips the run to finished and closes the event stream.
STAGE_DONE = "done"


# --------------------------------------------------------------------------- #
# Registries — bounded, on app.state (no module-level mutable state)
# --------------------------------------------------------------------------- #


@dataclass
class ClipState:
    clip_id: str
    tmp_dir: str
    video_path: str
    filename: str
    content_type: str
    size: int
    created: float = field(default_factory=time.monotonic)


@dataclass
class RunState:
    run_id: str
    clip_id: str
    started: float
    status: str = "processing"  # processing | done | error
    finished: bool = False
    result: Optional[dict] = None
    error: Optional[str] = None
    events: list[dict] = field(default_factory=list)


class ClipRegistry:
    """A bounded map of clip_id -> ClipState. Evicting a clip deletes its temp file."""

    def __init__(self, max_clips: int = 100) -> None:
        self._clips: dict[str, ClipState] = {}
        self._order: list[str] = []
        self._max = max_clips

    def create(self, clip: ClipState) -> ClipState:
        self._clips[clip.clip_id] = clip
        self._order.append(clip.clip_id)
        while len(self._order) > self._max:
            old = self._order.pop(0)
            gone = self._clips.pop(old, None)
            if gone is not None:
                shutil.rmtree(gone.tmp_dir, ignore_errors=True)
        return clip

    def get(self, clip_id: str) -> Optional[ClipState]:
        return self._clips.get(clip_id)

    def clear(self) -> None:
        for clip in self._clips.values():
            shutil.rmtree(clip.tmp_dir, ignore_errors=True)
        self._clips.clear()
        self._order.clear()


class RunRegistry:
    """A bounded map of run_id -> RunState (no disk; the clip owns the file)."""

    def __init__(self, max_runs: int = 200) -> None:
        self._runs: dict[str, RunState] = {}
        self._order: list[str] = []
        self._max = max_runs

    def create(self, run_id: str, clip_id: str) -> RunState:
        state = RunState(run_id=run_id, clip_id=clip_id, started=time.monotonic())
        self._runs[run_id] = state
        self._order.append(run_id)
        while len(self._order) > self._max:
            self._runs.pop(self._order.pop(0), None)
        return state

    def get(self, run_id: str) -> Optional[RunState]:
        return self._runs.get(run_id)


# --------------------------------------------------------------------------- #
# Result envelope — shaped to the frontend RunResult / StyledCaption types
# --------------------------------------------------------------------------- #

# Canonical style order the frontend expects.
_STYLE_ORDER = ("formal", "sarcastic", "humorous_tech", "humorous_non_tech")


def _result_envelope(run_id: str, task_id: str, clip_id: str, ledger: Any, result: Any) -> dict:
    """Build the JSON the frontend consumes: ledger + the four styled captions.

    ``video_url`` stays null: the frontend builds the absolute clip URL from ``clip_id``
    against its configured API base, which the backend cannot know reliably.
    """
    caps = result.captions  # dict[StyleName, StyledCaption]
    ordered = sorted(caps.values(), key=lambda c: _STYLE_ORDER.index(c.style.value)
                     if c.style.value in _STYLE_ORDER else 99)
    return {
        "run_id": run_id,
        "task_id": task_id,
        "clip_id": clip_id,
        "video_url": None,
        "ledger": ledger.model_dump(mode="json"),
        "captions": [c.model_dump(mode="json") for c in ordered],
        "candidates": {},  # rejected candidates are internal to the pipeline; omitted
        "degraded": bool(result.degraded),
        "error": result.error,
    }


def _emit_terminal(
    publish: Callable[..., None], *, ok: bool, detail: str,
    result: Optional[dict] = None, error: Optional[str] = None,
) -> None:
    """End a run. Emits ``Finished`` (the progress marker) then the terminal ``done`` event
    the frontend routes on. Used by *every* exit path — success, degraded, error, timeout —
    so the stream always closes and the UI never waits forever. A degraded run is a success:
    it produced captions, so ``ok=True`` and the result envelope is carried through.
    """
    publish(STAGE_FINISHED, "done" if ok else "error", detail, result=result, error=error)
    publish(STAGE_DONE, "completed" if ok else "error", detail, result=result, error=error)


# --------------------------------------------------------------------------- #
# The worker — reuses the existing pipeline; runs off the server loop
# --------------------------------------------------------------------------- #


async def _process(
    run_id: str,
    clip_id: str,
    task_id: str,
    video_path: str,
    styles: tuple[str, ...],
    publish: Callable[..., None],
) -> None:
    """Drive the existing pipeline for one clip, publishing the seven progress stages.

    Runs inside the worker thread's own event loop. Every ``publish`` marshals back to the
    server loop. Engine imports are local so the server can start without loading models.
    """
    import httpx  # noqa: PLC0415

    from claris.agent.config import AgentConfig  # noqa: PLC0415
    from claris.agent.main import resolve_providers  # noqa: PLC0415
    from claris.core.observability import NullSink  # noqa: PLC0415
    from claris.core.perception import PerceptionConfig, build_perception  # noqa: PLC0415
    from claris.core.pipeline import run_from_ledger  # noqa: PLC0415
    from claris.core.schema import ALL_STYLES, StyleName, Task  # noqa: PLC0415

    timeout_s = float(os.getenv("CLARIS_API_RUN_TIMEOUT_S", "600"))
    cfg = AgentConfig.from_env()
    sink = NullSink()  # the pipeline's fine-grained log; progress is published separately

    task_styles = tuple(StyleName(s) for s in styles) if styles else ALL_STYLES
    task = Task(task_id=task_id, video_path=video_path, styles=task_styles)

    async def _run() -> None:
        async with httpx.AsyncClient(timeout=httpx.Timeout(90.0)) as client:
            # 1) Discover models + build providers (reused from the batch agent verbatim).
            providers, roles = await resolve_providers(cfg, client=client)
            _log(logging.INFO, "providers resolved", run_id=run_id,
                 gen=roles.gen, vlm=roles.vlm, gemma=roles.gemma_path_used)
            perception_config = PerceptionConfig(
                vision_model=roles.vlm or PerceptionConfig().vision_model
            )

            # 2) Perception: video -> PerceptionBundle (textual ledger + keyframes).
            publish(STAGE_FRAMES, "active", "Probing container and sampling keyframes")
            publish(STAGE_SPEECH, "active", "Transcribing audio")
            publish(STAGE_OCR, "active", "Reading on-screen text")
            publish(STAGE_SCENE, "active", "Selecting keyframes")
            bundle = await build_perception(task, perception_config, sink=sink)
            ledger = bundle.ledger
            flags = ledger.modality_flags
            n = lambda kind: sum(1 for it in ledger.items if it.kind.value == kind)  # noqa: E731
            publish(STAGE_FRAMES, "done",
                    f"{len(ledger.items)} evidence items · coverage {ledger.coverage:.0%}")
            publish(STAGE_SPEECH, "done",
                    f"{n('speech')} utterances" if flags.has_speech
                    else ("no speech detected" if flags.is_silent else "no audio track"))
            publish(STAGE_OCR, "done",
                    f"{n('ocr')} text regions" if flags.has_ocr else "no on-screen text")
            publish(STAGE_SCENE, "done", f"{len(bundle.keyframes)} keyframes selected")

            # 3) One multimodal reasoning call: ledger + keyframes -> TaskResult.
            publish(STAGE_CAPTION, "active",
                    f"Reasoning over the clip and writing {len(task_styles)} styles")
            result = await run_from_ledger(
                ledger, task, providers, keyframes=bundle.keyframes,
                perception_config=perception_config, sink=sink, run_id=run_id,
            )
            publish(STAGE_CAPTION, "done",
                    "degraded" if result.degraded else "4 captions written")

            envelope = _result_envelope(run_id, task_id, clip_id, ledger, result)
            # Degraded is still a completed run with captions — end it as a success.
            _emit_terminal(publish, ok=True, detail="Captioning complete", result=envelope)
            _log(logging.INFO, "run complete", run_id=run_id, degraded=result.degraded)

    try:
        await asyncio.wait_for(_run(), timeout=timeout_s)
    except Exception as exc:  # noqa: BLE001 — one run's failure must not crash the server
        _log(logging.ERROR, "run failed", run_id=run_id, error=repr(exc))
        _emit_terminal(publish, ok=False, detail=f"Processing failed: {exc}", error=str(exc))


# --------------------------------------------------------------------------- #
# App factory
# --------------------------------------------------------------------------- #


@asynccontextmanager
async def _lifespan(app: FastAPI):
    yield
    clips = getattr(app.state, "clips", None)
    if clips is not None:
        clips.clear()  # remove any temp clip files on shutdown


def create_app() -> FastAPI:
    """Build and return the FastAPI application (used by ``--factory``)."""
    if not any(isinstance(h.formatter, _JsonFormatter) for h in _LOG.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(_JsonFormatter())
        _LOG.addHandler(handler)
        _LOG.setLevel(os.getenv("CLARIS_LOG_LEVEL", "INFO").upper())
        _LOG.propagate = False

    app = FastAPI(title="CLARIS API", version="1.0.0", lifespan=_lifespan)
    app.state.clips = ClipRegistry()
    app.state.runs = RunRegistry()

    origins = [o.strip() for o in os.getenv("CLARIS_CORS_ORIGINS", "*").split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=False,  # no cookies/auth; wildcard origins stay valid
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ----- internal helpers (shared by the /api routes and the aliases) ---- #

    def _publisher(loop: asyncio.AbstractEventLoop, run_id: str,
                   started: float) -> Callable[..., None]:
        """A thread-safe progress publisher bound to one run and the server loop."""
        runs: RunRegistry = app.state.runs

        def _append(event: dict) -> None:  # runs on the server loop
            state = runs.get(run_id)
            if state is None:
                return
            state.events.append(event)
            # Completion is keyed on the terminal "done" event (not "Finished"), so the
            # stream never breaks before "done" has been appended and sent.
            if event.get("stage") == STAGE_DONE:
                state.finished = True
                state.status = "error" if event.get("status") == "error" else "completed"
                state.result = event.get("result")
                state.error = event.get("error")

        def publish(stage: str, status: str, detail: str = "", **extra: Any) -> None:
            event = {"stage": stage, "status": status,
                     "t": round(time.monotonic() - started, 2), "detail": detail, **extra}
            loop.call_soon_threadsafe(_append, event)

        return publish

    async def _drive(run_id: str, clip_id: str, task_id: str, video_path: str,
                     styles: tuple[str, ...], publish: Callable[..., None]) -> None:
        """Run the (blocking) pipeline in a worker thread. The clip file is left in place so
        /api/clips/{clip_id}/video keeps working; cleanup happens on clip eviction."""
        try:
            def _worker() -> None:
                asyncio.run(_process(run_id, clip_id, task_id, video_path, styles, publish))

            await asyncio.to_thread(_worker)
        except Exception as exc:  # noqa: BLE001 — never let a run take down the server
            _log(logging.ERROR, "driver error", run_id=run_id, error=repr(exc))
            _emit_terminal(publish, ok=False, detail=f"Processing failed: {exc}", error=str(exc))

    async def _save_clip(file: UploadFile) -> tuple[Optional[ClipState], Optional[JSONResponse]]:
        """Validate + persist an uploaded video, register it, and return the ClipState."""
        name = file.filename or "clip.mp4"
        content_type = file.content_type or ""
        looks_video = content_type.startswith("video/") or name.lower().endswith(
            (".mp4", ".mov", ".m4v", ".webm")
        )
        if not looks_video:
            return None, JSONResponse(
                status_code=400,
                content={"error": f"Expected a video file (MP4); got '{name}' ({content_type})."},
            )

        clip_id = f"clip_{uuid.uuid4().hex[:12]}"
        tmp_dir = tempfile.mkdtemp(prefix="claris_clip_")
        dest = Path(tmp_dir) / f"{clip_id}.mp4"
        try:
            size = 0
            with dest.open("wb") as out:
                while chunk := await file.read(1 << 20):  # 1 MiB chunks
                    size += len(chunk)
                    out.write(chunk)
        except Exception as exc:  # noqa: BLE001
            shutil.rmtree(tmp_dir, ignore_errors=True)
            _log(logging.ERROR, "upload save failed", clip_id=clip_id, error=repr(exc))
            return None, JSONResponse(status_code=500, content={"error": "Failed to save upload."})
        finally:
            await file.close()

        if size == 0:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            return None, JSONResponse(status_code=400, content={"error": "Uploaded file is empty."})

        clip = app.state.clips.create(ClipState(
            clip_id=clip_id, tmp_dir=tmp_dir, video_path=str(dest),
            filename=name, content_type=content_type or "video/mp4", size=size,
        ))
        _log(logging.INFO, "clip stored", clip_id=clip_id, filename=name, bytes=size)
        return clip, None

    def _start_run(clip: ClipState) -> str:
        """Kick off the pipeline for a stored clip; returns the new run_id."""
        run_id = f"run_{uuid.uuid4().hex[:12]}"
        state = app.state.runs.create(run_id, clip.clip_id)
        loop = asyncio.get_running_loop()
        publish = _publisher(loop, run_id, state.started)
        publish(STAGE_UPLOAD, "done", f"Received {clip.filename} ({clip.size / 1_048_576:.1f} MB)")
        task_id = Path(clip.filename).stem or run_id
        loop.create_task(
            _drive(run_id, clip.clip_id, task_id, clip.video_path, _STYLE_ORDER, publish)
        )
        _log(logging.INFO, "run started", run_id=run_id, clip_id=clip.clip_id)
        return run_id

    def _result_response(run_id: str) -> JSONResponse:
        state = app.state.runs.get(run_id)
        if state is None:
            return JSONResponse(status_code=404, content={"error": "unknown run_id"})
        if not state.finished:
            return JSONResponse(status_code=202, content={"run_id": run_id, "status": "processing"})
        if state.error and state.result is None:
            return JSONResponse(status_code=200,
                                content={"run_id": run_id, "status": "error", "error": state.error})
        return JSONResponse(status_code=200, content=state.result)

    async def _stream_events(websocket: WebSocket, run_id: str) -> None:
        """Replay history then stream live progress until Finished."""
        await websocket.accept()
        state = app.state.runs.get(run_id)
        if state is None:
            # Terminal "done" (error) so the client stops waiting instead of hanging open.
            await websocket.send_json(
                {"stage": STAGE_DONE, "status": "error", "t": 0.0,
                 "detail": "unknown run_id", "error": "unknown run_id"}
            )
            await websocket.close()
            return

        idx = 0
        try:
            while True:
                while idx < len(state.events):
                    await websocket.send_json(state.events[idx])
                    idx += 1
                if state.finished:
                    break
                await asyncio.sleep(0.05)
        except WebSocketDisconnect:
            return
        finally:
            try:
                await websocket.close()
            except RuntimeError:
                pass

    # ----- routes: the frontend contract ----------------------------------- #

    @app.get("/")
    async def root() -> dict:
        return {"service": "claris-api", "status": "ok"}

    @app.get("/health")
    async def health() -> dict:
        # Deliberately trivial: no dependencies, no engine imports. Answers before any model
        # is loaded so a platform health check always succeeds.
        return {"status": "ok"}

    @app.exception_handler(Exception)
    async def _unhandled(_request, exc: Exception) -> JSONResponse:
        # Last-resort net: any unexpected error becomes a clean JSON 500 rather than a
        # dropped connection, so a single bad request can never take the process down.
        _log(logging.ERROR, "unhandled request error", error=repr(exc))
        return JSONResponse(status_code=500, content={"error": "internal server error"})

    @app.post("/api/clips")
    async def create_clip(file: UploadFile = File(...)) -> JSONResponse:
        """Accept one MP4 and store it. Processing starts on POST /api/clips/{id}/run."""
        clip, err = await _save_clip(file)
        if err is not None:
            return err
        assert clip is not None
        return JSONResponse(status_code=201, content={"clip_id": clip.clip_id})

    @app.post("/api/clips/{clip_id}/run")
    async def start_run(clip_id: str) -> JSONResponse:
        """Start the pipeline for a previously-uploaded clip."""
        clip = app.state.clips.get(clip_id)
        if clip is None:
            return JSONResponse(status_code=404, content={"error": "unknown clip_id"})
        return JSONResponse(status_code=202, content={"run_id": _start_run(clip)})

    @app.get("/api/clips/{clip_id}/video")
    async def get_video(clip_id: str):
        """Serve the stored clip (supports range requests for scrubbing)."""
        clip = app.state.clips.get(clip_id)
        if clip is None or not Path(clip.video_path).exists():
            return JSONResponse(status_code=404, content={"error": "unknown clip_id"})
        return FileResponse(clip.video_path, media_type="video/mp4", filename=clip.filename)

    @app.get("/api/fetch-video")
    async def fetch_video(url: str):
        """Server-side proxy for a public direct video URL the browser can't fetch due to CORS.

        Downloads server-side (no browser CORS applies) and streams the bytes back same-origin
        so the normal upload flow can continue unchanged. Direct .mp4/.webm only; share and
        streaming pages are rejected with a clear message.
        """
        import ipaddress  # noqa: PLC0415
        import re  # noqa: PLC0415
        import socket  # noqa: PLC0415
        from urllib.parse import urlparse  # noqa: PLC0415

        import httpx  # noqa: PLC0415

        MAX_BYTES = 500 * 1024 * 1024  # hard cap on the proxied download

        def _non_public(target: str) -> Optional[str]:
            # Not an open proxy: only fetch names that resolve to public IPs (block SSRF to
            # localhost / private / link-local / reserved ranges and cloud metadata).
            host = urlparse(target).hostname
            if not host:
                return "invalid host"
            try:
                infos = socket.getaddrinfo(host, None)
            except socket.gaierror:
                return "host did not resolve"
            for info in infos:
                ip = ipaddress.ip_address(info[4][0])
                if not ip.is_global or ip.is_multicast:
                    return "non-public address"
            return None

        if not re.match(r"^https?://", url, re.IGNORECASE):
            return JSONResponse(status_code=400, content={"error": "Provide a direct http(s) video URL."})
        low = url.lower()
        blocked = ("youtube.com", "youtu.be", "drive.google.com", "instagram.com",
                   "tiktok.com", "facebook.com", "vimeo.com/")
        if any(h in low for h in blocked):
            return JSONResponse(status_code=400, content={
                "error": "That's a share/streaming page, not a direct video file. "
                         "Paste a direct .mp4 or .webm link."})
        if _non_public(url) is not None:
            return JSONResponse(status_code=400, content={"error": "That host is not a public address."})

        client = httpx.AsyncClient(follow_redirects=True, timeout=httpx.Timeout(60.0))
        try:
            resp = await client.send(client.build_request("GET", url), stream=True)
        except Exception as exc:  # noqa: BLE001
            await client.aclose()
            return JSONResponse(status_code=502,
                                content={"error": f"Could not reach that URL ({exc.__class__.__name__})."})

        async def _reject(status: int, message: str) -> JSONResponse:
            await resp.aclose()
            await client.aclose()
            return JSONResponse(status_code=status, content={"error": message})

        # Redirects may land on an internal address; validate the final URL too.
        if _non_public(str(resp.url)) is not None:
            return await _reject(400, "That host is not a public address.")

        ctype = resp.headers.get("content-type", "")
        is_video = ctype.startswith("video/") or low.split("?")[0].endswith(
            (".mp4", ".mov", ".m4v", ".webm"))
        if resp.status_code >= 400 or not is_video:
            return await _reject(
                400, f"URL did not return a direct video (status {resp.status_code}, "
                     f"type {ctype or 'unknown'}).")

        clen = resp.headers.get("content-length")
        if clen and clen.isdigit() and int(clen) > MAX_BYTES:
            return await _reject(413, f"Video exceeds the {MAX_BYTES // (1 << 20)} MB limit.")

        async def _stream():
            total = 0
            try:
                async for chunk in resp.aiter_bytes(1 << 16):
                    total += len(chunk)
                    if total > MAX_BYTES:  # cap servers that lie about / omit content-length
                        break
                    yield chunk
            finally:
                await resp.aclose()
                await client.aclose()

        return StreamingResponse(_stream(), media_type=ctype or "video/mp4")

    @app.get("/api/runs/{run_id}")
    async def get_run(run_id: str) -> JSONResponse:
        return _result_response(run_id)

    @app.websocket("/api/runs/{run_id}/events")
    async def run_events(websocket: WebSocket, run_id: str) -> None:
        await _stream_events(websocket, run_id)

    # ----- routes: compatibility aliases ----------------------------------- #

    @app.post("/upload")
    async def upload(file: UploadFile = File(...)) -> JSONResponse:
        """One-shot alias: store the clip and start the run in a single call."""
        clip, err = await _save_clip(file)
        if err is not None:
            return err
        assert clip is not None
        return JSONResponse(status_code=202,
                            content={"run_id": _start_run(clip), "status": "processing"})

    @app.get("/result/{run_id}")
    async def result_alias(run_id: str) -> JSONResponse:
        return _result_response(run_id)

    @app.websocket("/ws/{run_id}")
    async def ws_alias(websocket: WebSocket, run_id: str) -> None:
        await _stream_events(websocket, run_id)

    return app


# A module-level ASGI app so the service starts under *any* conventional command —
# ``uvicorn claris.api.main:app`` — not only the ``create_app --factory`` form. Without this,
# a platform that auto-detects the entrypoint (or a start command that drops ``--factory``, or
# gunicorn, or an older uvicorn that doesn't auto-detect factories) calls the factory itself
# as the ASGI app: startup "succeeds", then every request invokes ``create_app(scope, receive,
# send)``, which raises — surfacing as a 502 at the proxy. Building the app here is cheap:
# create_app() does no heavy work and imports no ML, so this module stays light and /health
# needs nothing loaded.
app = create_app()


__all__ = ["create_app", "app"]
