"""ASR stage: audio bytes -> transcript text via local faster-whisper."""
import io
from functools import lru_cache

from app import config


@lru_cache(maxsize=1)
def _model():
    # Imported and loaded lazily so tests and tooling don't pay the model-load cost.
    from faster_whisper import WhisperModel

    return WhisperModel(config.WHISPER_MODEL, device=config.WHISPER_DEVICE, compute_type=config.WHISPER_COMPUTE_TYPE)


def transcribe(audio_bytes: bytes) -> str:
    """Return the transcript, or "" if the audio contains no usable speech.

    `vad_filter` drops non-speech regions before decoding, and segments Whisper itself
    flags as probable non-speech are discarded - this stops Whisper from inventing text
    (e.g. "Thank you for watching") on silent or noise-only recordings.
    """
    segments, _info = _model().transcribe(io.BytesIO(audio_bytes), vad_filter=True)
    kept = [s.text for s in segments if s.no_speech_prob < config.NO_SPEECH_THRESHOLD]
    return "".join(kept).strip()
