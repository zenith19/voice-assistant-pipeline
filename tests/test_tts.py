import shutil

import pytest

from app.pipeline import tts

pytestmark = pytest.mark.skipif(shutil.which("espeak-ng") is None, reason="espeak-ng not installed")


def test_synthesize_returns_wav():
    assert tts.synthesize("hello").startswith(b"RIFF")


def test_text_starting_with_dash_is_spoken_not_parsed_as_flag():
    # With the old argv-based call, "--help" printed usage text instead of audio.
    assert tts.synthesize("--help").startswith(b"RIFF")
