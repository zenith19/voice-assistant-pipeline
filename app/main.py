"""FastAPI app wiring the ASR -> LLM(+tools) -> TTS pipeline together."""
import base64
from typing import Annotated

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app import config
from app.pipeline import asr, llm, tts

app = FastAPI(title="Voice Assistant Pipeline")
app.mount("/static", StaticFiles(directory=config.STATIC_DIR), name="static")

conversation_history: list[dict] = []

NO_SPEECH_REPLY = "Sorry, I didn't catch that. Could you say it again?"


@app.get("/")
def index():
    return FileResponse(config.STATIC_DIR / "index.html")


# Plain `def` (not `async def`): ASR, the LLM call, and TTS are all blocking, so FastAPI
# runs this handler in its threadpool instead of stalling the event loop.
@app.post("/chat")
def chat(audio: Annotated[UploadFile, File()]):
    audio_bytes = audio.file.read()

    transcript = asr.transcribe(audio_bytes)
    if not transcript:
        # Silence / noise only: don't send an empty turn to the LLM or into history.
        turn = {"reply": NO_SPEECH_REPLY, "actions": []}
    else:
        turn = llm.run_turn(conversation_history, transcript)
    speech_bytes = tts.synthesize(turn["reply"])

    return {
        "transcript": transcript,
        "reply": turn["reply"],
        "actions": turn["actions"],
        "audio_base64": base64.b64encode(speech_bytes).decode("ascii"),
    }


@app.post("/reset")
def reset():
    conversation_history.clear()
    return {"status": "conversation reset"}
