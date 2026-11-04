# Voice Assistant Pipeline

A small ASR -> LLM (with tool-calling) -> TTS voice assistant, built to learn Claude Code hands-on.

Talk into your browser mic, the assistant transcribes what you said, decides whether it needs
to call a tool (time, calculator, weather, reminders) to answer, and speaks the reply back.

## Architecture

```
mic (browser) -> POST /chat -----------------------------------> speech reply
                     |
                     v
             app/pipeline/asr.py   (local faster-whisper: audio -> text)
                     v
             app/pipeline/llm.py   (local Ollama model: decides + calls tools in app/pipeline/tools.py)
                     v
             app/pipeline/tts.py   (local espeak-ng: text -> audio)
```

Runs fully locally, no API key or billing required: ASR via `faster-whisper` (CPU), the LLM
via [Ollama](https://ollama.com) serving `qwen2.5:1.5b`, and TTS via `espeak-ng`.

- `app/main.py` - FastAPI app, wires the three stages together in the `/chat` endpoint
- `app/pipeline/asr.py` - speech-to-text
- `app/pipeline/llm.py` - the agent loop: sends the transcript + available tools to the model,
  executes any tool calls it asks for, loops until it has a final spoken reply
- `app/pipeline/tools.py` - the demo tools (`get_current_time`, `calculate`, `get_weather`,
  `set_reminder`) and their function-calling schemas
- `app/pipeline/tts.py` - text-to-speech
- `app/static/index.html` - mic button UI (browser `MediaRecorder` + `fetch`)

## Setup

Prerequisites (installed once, outside the venv):

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen2.5:1.5b
sudo apt install -y espeak-ng
```

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Settings (model names, Ollama URL, Whisper size, no-speech threshold) are read from environment
variables with local defaults - see `app/config.py`. For example, `LLM_MODEL=qwen2.5:3b` swaps
the model without code changes.

## Run

```bash
uvicorn app.main:app --reload
```

Open http://localhost:8000, allow mic access, click **Record**, speak, click **Stop**. The page
shows the transcript, any tools the assistant called, its reply, and plays the spoken response.

Try things like:
- "What time is it?"
- "What's 23 times 17?"
- "What's the weather in Berlin?"
- "Remind me to call mom in 20 minutes."

Click **Reset conversation** to clear the assistant's memory of the current session.

## Tests

```bash
pip install -r requirements-dev.txt
ruff check .
pytest
```

The unit tests stub the ASR and LLM stages (a fake OpenAI-compatible client replays scripted
model responses), so they run without Ollama or a Whisper model. They cover the tool-calling
loop's failure paths (malformed tool JSON, wrong arguments, unknown tools, running out of tool
rounds), silence handling, and that concurrent requests don't block each other. CI runs the same
checks on every push and pull request.

## Extending

To add a new action, add a function + JSON schema entry in `app/pipeline/tools.py` and register
it in `DISPATCH` - the LLM stage picks it up automatically, no other changes needed.
