# CLARIS

**Multi-style video captioning that stays true to what's on screen.**

CLARIS turns a short video clip into four ready-to-use captions — each in a distinct
voice — while keeping every caption faithful to what actually happens in the footage.
Drop in a 30-second to two-minute clip and get back a formal caption, a sarcastic one,
and two flavors of humor, all describing the same moment through different lenses.

## What CLARIS does

- **Four captions, four voices.** Every clip is captioned in four styles — formal,
  sarcastic, humorous (tech), and humorous (everyday) — so you can match the caption to
  the channel, audience, or mood.
- **Grounded in the footage.** Captions describe what is genuinely seen and heard in the
  clip, not generic filler.
- **Distinct by design.** The four styles stay recognizably different from one another
  instead of collapsing into slight rewordings of the same sentence.
- **Multimodal understanding.** CLARIS draws on speech, on-screen text, ambient audio, and
  the visuals of a clip to understand what it is about.

## Built with

CLARIS brings together leading open models and infrastructure:

| Capability | Powered by |
|---|---|
| Language & vision understanding | Gemma |
| Speech recognition | Whisper |
| On-screen text recognition | PaddleOCR |
| Video & frame processing | OpenCV |
| Model inference | Fireworks AI |

## Getting started

### Prerequisites

- Python 3.11 or newer
- [FFmpeg](https://ffmpeg.org/) available on your `PATH`
- A [Fireworks AI](https://fireworks.ai/) API key

### Setup

1. Install CLARIS and its dependencies.
2. Provide your Fireworks AI API key through a `FIREWORKS_API_KEY` environment variable (a
   local `.env` file is supported).
3. Point CLARIS at a video clip to receive its four captions.

## License

CLARIS is released under the MIT License. See [LICENSE](LICENSE) for details.

## Contact

Questions, feedback, or partnership inquiries: **sashikumaaran@gmail.com**
