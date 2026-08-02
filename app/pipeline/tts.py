"""TTS stage: reply text -> spoken audio bytes via OpenAI TTS."""
from openai import OpenAI


def synthesize(client: OpenAI, text: str, voice: str = "alloy") -> bytes:
    response = client.audio.speech.create(
        model="tts-1",
        voice=voice,
        input=text,
    )
    return response.content
