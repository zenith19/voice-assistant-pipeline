"""FastAPI app wiring the ASR -> LLM(+tools) -> TTS pipeline together."""
import base64

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.pipeline import asr, llm, tts

app = FastAPI(title="Voice Assistant Pipeline")
app.mount("/static", StaticFiles(directory="app/static"), name="static")

conversation_history: list[dict] = []


@app.get("/")
def index():
    return FileResponse("app/static/index.html")


@app.post("/chat")
async def chat(audio: UploadFile = File(...)):
    audio_bytes = await audio.read()

    transcript = asr.transcribe(audio_bytes)
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
