"""TTS stage: reply text -> spoken audio bytes via local espeak-ng."""
import subprocess

from app import config


def synthesize(text: str) -> bytes:
    # Text goes in on stdin, not argv, so a reply starting with "-" can't be parsed as a flag.
    result = subprocess.run(
        [config.ESPEAK_BINARY, "--stdout"],
        input=text.encode("utf-8"),
        capture_output=True,
        check=True,
    )
    return result.stdout
