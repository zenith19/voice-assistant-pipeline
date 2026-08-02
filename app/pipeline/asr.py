"""ASR stage: audio bytes -> transcript text via OpenAI Whisper."""
from openai import OpenAI


def transcribe(client: OpenAI, filename: str, audio_bytes: bytes, content_type: str) -> str:
    result = client.audio.transcriptions.create(
        model="whisper-1",
        file=(filename, audio_bytes, content_type),
    )
    return result.text
