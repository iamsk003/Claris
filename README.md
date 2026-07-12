# CLARIS AI

**Multi-style video captioning that stays true to what's on screen.**

CLARIS AI turns a short video clip into four ready-to-use captions — each in a distinct
voice — while keeping every caption faithful to what actually happens in the footage. Drop
in a 30-second to two-minute clip and get back a formal caption, a sarcastic one, and two
flavors of humor, all describing the same moment through different lenses.

## Overview

CLARIS AI reads a clip the way a person would: it watches the keyframes, listens to the
speech, reads any on-screen text, and notes the ambient audio and motion. Those signals are
collected into a single, timestamped **evidence ledger**, and one multimodal reasoning pass
turns that evidence into a faithful description and four styled captions. Because every
caption is written from the same grounded evidence, the four voices stay distinct without
drifting from what the video actually shows.

## Features

- **Four captions, four voices.** Every clip is captioned in four styles — formal,
  sarcastic, tech humor, and everyday humor — so you can match the caption to the channel,
  audience, or mood.
- **Grounded in the footage.** Captions describe what is genuinely seen and heard in the
  clip, drawn from timestamped evidence rather than generic filler.
- **Distinct by design.** The four styles share the same facts but stay recognizably
  different instead of collapsing into slight rewordings of one sentence.
- **Multimodal understanding.** Speech, on-screen text, ambient audio, motion, and the
  visuals of a clip are combined into one evidence object before a caption is written.
- **Traceable.** Each caption is tied back to the evidence that supports it, so you can see
  why the model said what it said.
- **One reasoning call.** A single multimodal pass produces the summary, timeline, and all
  four captions — fast, and inexpensive to run.

## Architecture

CLARIS AI is one engine behind a batch agent, an HTTP/WebSocket API, and a web app. The core
flow is a straight line:

```
Video
  ↓
Perception            (keyframes · speech · OCR · audio events · motion)
  ↓
Evidence Ledger       (immutable, timestamped, per-modality facts)
  ↓
Single Multimodal Reasoning   (one call: keyframes + structured evidence → JSON)
  ↓
Four Caption Styles   (formal · sarcastic · tech humor · everyday humor)
```

Perception samples a handful of representative keyframes and runs local speech, OCR, audio,
and motion detectors, assembling them into a read-only evidence ledger. A single multimodal
reasoning call receives the ordered keyframes plus the structured evidence and returns one
JSON object — a summary, a chronological event list, and the four styled captions — which is
mapped to the final result. There is no per-frame captioning and no multi-pass verification
loop; grounding comes from constraining the one call to the supplied evidence.

## Technology stack

| Layer | Technology |
|---|---|
| Multimodal reasoning | Model-agnostic multimodal LLM served via Fireworks AI |
| Speech recognition | faster-whisper |
| On-screen text | PaddleOCR |
| Keyframes & motion | OpenCV · PySceneDetect |
| Audio events | librosa |
| Core & agent | Python 3.11 · Pydantic v2 |
| API | FastAPI · WebSockets |
| Web app | Vite · React · TypeScript · Tailwind CSS |
| Packaging & deploy | Docker · Railway · Vercel |

## Installation

### Prerequisites

- Python 3.11 or newer
- [FFmpeg](https://ffmpeg.org/) available on your `PATH`
- A [Fireworks AI](https://fireworks.ai/) API key

### Setup

```bash
# 1. Install dependencies (uv shown; pip works too)
uv sync

# 2. Provide your Fireworks AI key (a local .env file is supported)
cp .env.example .env
echo "FIREWORKS_API_KEY=fw_your_key_here" >> .env

# 3. Run the batch agent over /input/tasks.json
CLARIS_INPUT=input/tasks.json CLARIS_OUTPUT=output/results.json \
CLARIS_RUN_LOG=output/run_log.jsonl python -m claris.agent.main

# 4. (Optional) Run the API and web app for the interactive demo
uvicorn claris.api.main:app --host 0.0.0.0 --port 8000
cd web && npm install && npm run dev   # set VITE_API_URL to the API
```

## Docker usage

Two images: the batch agent (root `Dockerfile`, the default command runs the agent and
exits) and the API (`Dockerfile.web`).

```bash
# Pull the published agent image
docker pull ghcr.io/iamsk003/claris-ai:latest

# Run the agent on a mounted input/output pair (needs a Fireworks key)
docker run --rm \
  -e FIREWORKS_API_KEY=$FIREWORKS_API_KEY \
  -v "$PWD/input:/input:ro" -v "$PWD/output:/output" \
  ghcr.io/iamsk003/claris-ai:latest

# Or build locally
docker build -t claris-agent .                 # the agent
docker build -f Dockerfile.web -t claris-api . # the API
```

## Railway deployment

The API deploys to [Railway](https://railway.app/) from `Dockerfile.web`, which serves the
module-level app and binds Railway's `$PORT`:

```bash
uvicorn claris.api.main:app --host 0.0.0.0 --port ${PORT}
```

Set `FIREWORKS_API_KEY` in the Railway service variables. Point the web app's `VITE_API_URL`
at the deployed API URL (the web app builds as a static site and deploys to Vercel).

## Example output

`output/results.json` — one entry per clip, four captions per entry:

```json
{
  "run_id": "agent_20260712T120000",
  "results": [
    {
      "task_id": "cooking",
      "captions": {
        "formal": "A person prepares bread at a stovetop, then plates the finished loaf.",
        "sarcastic": "Groundbreaking footage: someone makes bread and lives to plate it.",
        "humorous_tech": "Dough compiled, proofing passed, loaf shipped to the plate — no rollback needed.",
        "humorous_non_tech": "Carbs incoming, and honestly, no regrets."
      }
    }
  ]
}
```

## Screenshots

The web app renders each run as four captions beside the source video and a timestamped
evidence timeline, with per-caption evidence you can trace back to the footage.

| View | Preview |
|---|---|
| Upload | _add `docs/screenshots/upload.png`_ |
| Processing | _add `docs/screenshots/processing.png`_ |
| Results | _add `docs/screenshots/results.png`_ |

## License

CLARIS AI is released under the MIT License. See [LICENSE](LICENSE) for details.
