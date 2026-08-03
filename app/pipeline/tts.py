"""TTS stage: reply text -> spoken audio bytes via local espeak-ng."""
import subprocess


def synthesize(text: str) -> bytes:
    result = subprocess.run(
        ["espeak-ng", "--stdout", text],
        capture_output=True,
        check=True,
    )
    return result.stdout
