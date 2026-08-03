"""ASR stage: audio bytes -> transcript text via local faster-whisper."""
import io

from faster_whisper import WhisperModel

_model = WhisperModel("small", device="cpu", compute_type="int8")


def transcribe(audio_bytes: bytes) -> str:
    segments, _info = _model.transcribe(io.BytesIO(audio_bytes))
    return "".join(segment.text for segment in segments).strip()
