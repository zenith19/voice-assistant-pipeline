"""One full voice turn (ASR -> LLM with tools -> TTS) with per-stage timings.

Shared by the `/chat` endpoint and the offline evaluation, so both measure the same code path.
"""
import time

from . import asr, llm, tts

NO_SPEECH_REPLY = "Sorry, I didn't catch that. Could you say it again?"


def _ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 1)


def respond(history: list[dict], transcript: str, synthesize: bool = True) -> dict:
    """LLM + TTS for an already-transcribed utterance."""
    if not transcript:
        # Silence / noise only: don't send an empty turn to the LLM or into history.
        turn = {"reply": NO_SPEECH_REPLY, "actions": [], "timings": {"llm_rounds_ms": [], "tools_ms": 0.0}}
    else:
        turn = llm.run_turn(history, transcript)

    timings = {
        "llm_ms": round(sum(turn["timings"]["llm_rounds_ms"]), 1),
        "llm_rounds_ms": turn["timings"]["llm_rounds_ms"],
        "tools_ms": turn["timings"]["tools_ms"],
    }
    speech = b""
    if synthesize:
        started = time.perf_counter()
        speech = tts.synthesize(turn["reply"])
        timings["tts_ms"] = _ms(started)

    return {"transcript": transcript, "reply": turn["reply"], "actions": turn["actions"],
            "llm_called": bool(transcript), "audio": speech, "timings": timings}


def run_voice_turn(history: list[dict], audio_bytes: bytes, synthesize: bool = True) -> dict:
    started = time.perf_counter()
    transcript = asr.transcribe(audio_bytes)
    asr_ms = _ms(started)

    result = respond(history, transcript, synthesize=synthesize)
    result["timings"] = {"asr_ms": asr_ms, **result["timings"], "total_ms": _ms(started)}
    return result
