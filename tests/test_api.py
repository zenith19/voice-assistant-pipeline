import asyncio
import base64
import time

import httpx
import pytest
from fastapi.testclient import TestClient

from app import main
from app.pipeline import turn


@pytest.fixture(autouse=True)
def stub_stages(monkeypatch):
    main.conversation_history.clear()
    monkeypatch.setattr(turn.tts, "synthesize", lambda text: b"RIFFfake")
    monkeypatch.setattr(turn.llm, "run_turn", lambda history, text: {
        "reply": f"echo: {text}", "actions": [], "timings": {"llm_rounds_ms": [12.0], "tools_ms": 0.0}})


def post_audio(client):
    return client.post("/chat", files={"audio": ("input.webm", b"\x00" * 10, "audio/webm")})


def test_chat_returns_transcript_reply_and_audio(monkeypatch):
    monkeypatch.setattr(turn.asr, "transcribe", lambda audio: "hello")
    body = post_audio(TestClient(main.app)).json()

    assert body["transcript"] == "hello"
    assert body["reply"] == "echo: hello"
    assert base64.b64decode(body["audio_base64"]) == b"RIFFfake"
    timings = body["timings"]
    assert timings["llm_ms"] == 12.0
    assert {"asr_ms", "tts_ms", "total_ms", "llm_rounds_ms", "tools_ms"} <= timings.keys()


def test_silence_skips_llm(monkeypatch):
    monkeypatch.setattr(turn.asr, "transcribe", lambda audio: "")

    def fail(*args):
        raise AssertionError("LLM must not be called for empty transcripts")

    monkeypatch.setattr(turn.llm, "run_turn", fail)
    body = post_audio(TestClient(main.app)).json()
    assert body["reply"] == turn.NO_SPEECH_REPLY


def test_requests_do_not_block_each_other(monkeypatch):
    """Blocking stages run in the threadpool, so two slow turns overlap instead of queueing."""
    delay = 0.5

    def slow_transcribe(audio):
        time.sleep(delay)
        return "hello"

    monkeypatch.setattr(turn.asr, "transcribe", slow_transcribe)

    async def run_two():
        transport = httpx.ASGITransport(app=main.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            files = {"audio": ("input.webm", b"\x00", "audio/webm")}
            start = time.perf_counter()
            responses = await asyncio.gather(client.post("/chat", files=files),
                                             client.post("/chat", files=files))
            return time.perf_counter() - start, responses

    elapsed, responses = asyncio.run(run_two())
    assert all(r.status_code == 200 for r in responses)
    assert elapsed < 2 * delay * 0.9


def test_index_is_served_regardless_of_cwd(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    assert "Voice Assistant" in TestClient(main.app).get("/").text
