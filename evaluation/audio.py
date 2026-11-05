"""Audio conditions for the evaluation."""
import io
import wave

import numpy as np

TELEPHONY_RATE = 8000
_MU = 255.0


def mu_law_roundtrip(samples: np.ndarray) -> np.ndarray:
    """Encode float samples in [-1, 1] to 8-bit mu-law and decode them again (G.711-style)."""
    x = np.clip(samples, -1.0, 1.0)
    companded = np.sign(x) * np.log1p(_MU * np.abs(x)) / np.log1p(_MU)
    quantized = np.round((companded + 1) / 2 * _MU) / _MU * 2 - 1      # 256 levels
    return np.sign(quantized) * ((1 + _MU) ** np.abs(quantized) - 1) / _MU


def to_wav_bytes(samples: np.ndarray, rate: int) -> bytes:
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())
    return buffer.getvalue()


def telephony(audio_bytes: bytes) -> bytes:
    """Approximate a phone line: downsample to 8 kHz (drops everything above ~4 kHz) and
    quantize with 8-bit mu-law. No packet loss, jitter, or 300 Hz high-pass is simulated.
    The result is an 8 kHz WAV; the ASR stage resamples it back to 16 kHz as it would for a
    real telephony stream."""
    from faster_whisper.audio import decode_audio

    samples = decode_audio(io.BytesIO(audio_bytes), sampling_rate=TELEPHONY_RATE)
    return to_wav_bytes(mu_law_roundtrip(samples), TELEPHONY_RATE)
