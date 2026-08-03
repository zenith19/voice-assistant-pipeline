# Voice Assistant Pipeline — Project Notes

## What this is

A browser-based voice assistant: speak into the mic, it transcribes what you
said, runs it through an LLM that can call tools (time, calculator, weather,
reminders), and speaks the reply back. Originally scaffolded around the
OpenAI API (Whisper, gpt-4o-mini, TTS-1); now runs **entirely locally and
free**, no API key or billing required.

Built as a learning project for working hands-on with Claude Code.

## Current state

Working end-to-end on a local dev machine:

- Browser mic UI records audio and posts it to `/chat`.
- Audio is transcribed, sent to the LLM with a tool-calling loop, tool
  results are fed back to the model, and the final reply is synthesized to
  speech and played back in the browser.
- `/reset` clears the (single, in-memory, process-global) conversation
  history.
- Four demo tools: `get_current_time`, `calculate` (safe AST-based
  arithmetic, no `eval`), `get_weather` (mock data), `set_reminder`
  (in-memory, doesn't actually schedule anything).
- A stored-XSS bug in the mic UI (raw `innerHTML` interpolation of
  transcript/reply/tool output) was found and fixed by building DOM nodes
  with `textContent` instead.

### Known limitations

- **Small local LLM reliability**: `qwen2.5:1.5b` is small enough to run on
  a memory-constrained machine, but needs a tightly worded system prompt and
  low sampling temperature (`0.1`) to reliably choose tool calls over
  guessing/hallucinating (e.g. it initially fabricated fake dates instead of
  calling `get_current_time`). Even tuned, it's less reliable than a bigger
  model would be.
- **espeak-ng TTS voice** is robotic, not natural-sounding — fine for a demo,
  not for anything user-facing.
- **`faster-whisper` "small" model** — reasonable CPU-only accuracy/speed
  tradeoff, but not as accurate as larger Whisper models or the hosted
  OpenAI Whisper API, especially on non-standard voices/accents/noisy audio.
- **Conversation state is a single global in-process list** — no
  per-user/session isolation; fine for a single local user, not for
  multiple concurrent users.
- **No auth** on `/chat` or `/reset` — acceptable for local-only use.
- Dev/test machine has tight RAM (only ~1-2GB free of 15GB, swap often
  near-full), which is why the local model stack is intentionally small
  (Whisper `small`, Ollama `qwen2.5:1.5b`, `espeak-ng`) rather than the
  higher-quality options.

## Tech stack

| Stage | Tech | Notes |
|---|---|---|
| Web framework | FastAPI + Uvicorn | `app/main.py` |
| Frontend | Vanilla JS, browser `MediaRecorder` API | `app/static/index.html`, no build step |
| ASR (speech → text) | `faster-whisper`, model `small`, CPU, int8 | `app/pipeline/asr.py` |
| LLM + tool-calling | Ollama, serving `qwen2.5:1.5b`, via the OpenAI-compatible `/v1` API (`openai` Python client pointed at `http://127.0.0.1:11434/v1`) | `app/pipeline/llm.py` |
| TTS (text → speech) | `espeak-ng` CLI via `subprocess`, WAV output | `app/pipeline/tts.py` |
| Tools | Plain Python functions + JSON schemas, dispatched via a fixed dict (no dynamic/arbitrary function lookup) | `app/pipeline/tools.py` |

Nothing here needs internet access at runtime (aside from Ollama's initial
model pull and Whisper's initial model download from Hugging Face, both
one-time, cached locally afterward).

## Architecture

```
mic (browser) -> POST /chat -----------------------------------> speech reply
                     |
                     v
             app/pipeline/asr.py   (faster-whisper: audio -> text)
                     v
             app/pipeline/llm.py   (Ollama/qwen2.5:1.5b: decides + calls tools)
                     v                        |
                     |                app/pipeline/tools.py (tool implementations + schemas)
                     v
             app/pipeline/tts.py   (espeak-ng: text -> audio)
```

`app/main.py` is the only place that wires the three stages together (in the
`/chat` handler) and holds the process-global `conversation_history` list.
Each pipeline module is a thin, swappable stage with a narrow
bytes-in/text-out or text-in/bytes-out interface — that's what made it
possible to swap every stage from OpenAI's hosted API to local equivalents
without touching `main.py`'s control flow or `tools.py` at all.

## How to extend

**Add a new tool**: add a function + JSON schema entry to
`app/pipeline/tools.py`, register it in `DISPATCH`. `llm.py` picks it up
automatically — no other changes needed. (See README "Extending" section.)

**Improve reliability of the local LLM**:
- If more RAM becomes available, swap `MODEL` in `llm.py` to a larger Ollama
  model (e.g. `qwen2.5:7b`) for much better tool-calling reliability.
- Add few-shot examples to `SYSTEM_PROMPT` for tools the model still
  struggles with.
- Consider `tool_choice="required"` for turns you can detect are
  action-oriented, though this needs care to avoid forcing tool calls on
  pure chit-chat.

**Multi-user / session support**: replace the single global
`conversation_history` list with a session-keyed store (e.g. keyed by a
cookie or client-generated session id), since right now every browser tab
talking to the same server shares one conversation.

**Real reminders**: `set_reminder` currently just appends to an in-memory
list and does nothing else. Wiring it to an actual scheduler (e.g.
`apscheduler`) and a notification path (push, desktop notification, etc.)
would make it functional rather than a stub.

**Real weather**: swap `get_weather`'s mock dict for a real weather API
call (would reintroduce an external dependency/API key, unlike the rest of
this stack).

**Better TTS voice**: `espeak-ng` is free but robotic. A local
higher-quality alternative (e.g. Piper TTS) would sound much more natural
while staying free/offline; would need a new model download and a
`tts.py` rewrite, same shape as the current one.

**Streaming**: the whole pipeline currently runs as one blocking
request/response per turn (record → stop → wait → hear reply). Streaming
ASR partial transcripts and/or streaming the LLM's tokens back to the
browser as they're generated would make the assistant feel much more
responsive, at the cost of a more complex frontend (WebSocket instead of a
single `fetch`).

**Packaging the local setup**: Ollama + the model pull + `espeak-ng` are
currently manual one-time setup steps documented in the README. A
`docker-compose.yml` or setup script (see `/run-skill-generator` in Claude
Code) would make onboarding a new dev machine faster and more repeatable.
