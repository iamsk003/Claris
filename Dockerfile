# CLARIS batch agent image. Default CMD runs the agent and exits 0.
# The default CMD must not start a web server — that lives in Dockerfile.web.
#
# Build a linux/amd64 image with:
#   docker buildx build --platform linux/amd64 -t claris:latest .

# ---- build stage ------------------------------------------------------------
FROM python:3.11-slim AS build
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /app

# ffmpeg + libgl are needed by opencv/librosa/faster-whisper at import.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        build-essential \
        gcc \
        g++ \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock README.md ./
COPY claris ./claris
RUN uv sync --locked --no-dev

# Pre-download ALL model weights at BUILD time so the container needs no network for model
# downloads at runtime. sentence-transformers (gate_3) + faster-whisper (ASR) are baked into
# the image cache under HF_HOME.
ENV HF_HOME=/opt/hf-cache \
    SENTENCE_TRANSFORMERS_HOME=/opt/hf-cache \
    CLARIS_WHISPER_MODEL=base
# Hard preloads (no `|| true`): a successful build therefore GUARANTEES these caches are
# baked, which is what makes HF_HUB_OFFLINE safe in the runtime stage.
RUN uv run python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"
RUN uv run python -c "from faster_whisper import download_model; download_model('base')"

# PaddleOCR 2.x ignores any base-dir env and downloads its det/rec/angle-cls weights to its
# default home (~/.paddleocr == /root/.paddleocr in this stage). Warm that home so OCR needs
# no network at runtime. The verify line prints the resolved base dir into the build log so
# the copy target is confirmed, not assumed. Non-fatal on purpose: under x86 emulation on an
# arm build host paddle can SIGSEGV at import, and OCR degrades to empty — a native amd64
# build populates the cache; the mkdir keeps the runtime COPY valid either way.
RUN mkdir -p /root/.paddleocr \
 && uv run python -c "from paddleocr import PaddleOCR; PaddleOCR(use_angle_cls=True, lang='en', show_log=False)" || true
RUN uv run python -c "import paddleocr.paddleocr as p, os; d=p.BASE_DIR; print('PADDLE_BASE_DIR', d, 'exists', os.path.isdir(d), 'files', sum(len(f) for _,_,f in os.walk(d)))" || true

# ---- runtime stage ----------------------------------------------------------
FROM python:3.11-slim AS runtime
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=build /app/.venv /app/.venv
COPY --from=build /opt/hf-cache /opt/hf-cache
COPY --from=build /root/.paddleocr /root/.paddleocr
COPY claris ./claris
COPY eval ./eval

# HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE guarantee the preloaded caches are used and no
# request goes to HuggingFace at runtime. Safe because the build stage's HF preloads are
# hard (no `|| true`), so this image only exists if those caches were baked.
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/hf-cache \
    SENTENCE_TRANSFORMERS_HOME=/opt/hf-cache \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    CLARIS_WHISPER_MODEL=base \
    CLARIS_CACHE_DIR=/app/.claris_cache \
    CLARIS_LOG_DIR=/app/.claris_logs

# The caller mounts /input (ro) and /output and supplies a FIREWORKS_API_KEY. The agent
# discovers models, captions each task, and exits 0.
ENTRYPOINT ["python", "-m", "claris.agent.main"]
CMD []
