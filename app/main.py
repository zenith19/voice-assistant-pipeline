"""FastAPI app wiring the ASR -> LLM(+tools) -> TTS pipeline together."""
import base64
import json
import logging
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app import config
from app.pipeline import asr, turn

# One JSON line per turn on stdout, with per-stage latencies; easy to grep or load into pandas.
turn_log = logging.getLogger("voice.turns")
turn_log.setLevel(logging.INFO)
if not turn_log.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(message)s"))
    turn_log.addHandler(_handler)
    turn_log.propagate = False


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Load the Whisper model at startup instead of on the first request (cold start was ~37 s
    # on the dev machine, almost all of it model loading).
    if config.PRELOAD_MODELS:
        asr.preload()
    yield


app = FastAPI(title="Voice Assistant Pipeline", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=config.STATIC_DIR), name="static")

conversation_history: list[dict] = []


@app.get("/")
def index():
    return FileResponse(config.STATIC_DIR / "index.html")


# Plain `def` (not `async def`): ASR, the LLM call, and TTS are all blocking, so FastAPI
# runs this handler in its threadpool instead of stalling the event loop.
@app.post("/chat")
def chat(audio: Annotated[UploadFile, File()]):
    result = turn.run_voice_turn(conversation_history, audio.file.read())

    turn_log.info(json.dumps({
        "event": "turn",
        "provider": config.LLM_PROVIDER,
        "model": config.LLM_MODEL,
        "llm_called": result["llm_called"],
        "tools": [a["tool"] for a in result["actions"]],
        **result["timings"],
    }))

    return {
        "transcript": result["transcript"],
        "reply": result["reply"],
        "actions": result["actions"],
        "timings": result["timings"],
        "audio_base64": base64.b64encode(result["audio"]).decode("ascii"),
    }


@app.post("/reset")
def reset():
    conversation_history.clear()
    return {"status": "conversation reset"}
