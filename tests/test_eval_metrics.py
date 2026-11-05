import io
import wave

import numpy as np
import pytest

from evaluation import audio, metrics


@pytest.mark.parametrize("text, expected", [
    ("What's 23 times 17?", "what is 23 times 17"),
    ("what is twenty-three times seventeen", "what is 23 times 17"),
    ("What is one hundred and forty four divided by twelve?", "what is 144 divided by 12"),
    ("Remind me to, um, call the dentist.", "remind me to call the dentist"),
    ("Calculate 15% of 80.", "calculate 15 percent of 80"),
    ("It's 1,000 minus 357.", "it is 1000 minus 357"),
    ("Split 22.5 euros.", "split 22.5 euros"),
    ("What's the weather in... Munich?", "what is the weather in munich"),
    ("two thousand and five", "2005"),
    ("", ""),
])
def test_normalize(text, expected):
    assert metrics.normalize(text) == expected


def test_corpus_wer_ignores_formatting_and_skips_empty_references():
    refs = ["What's 23 times 17?", "", "Remind me to call mom."]
    hyps = ["what is twenty three times seventeen", "thank you for watching", "Remind me to call Tom."]
    # 1 substitution (mom -> tom) over 5 + 5 reference words; the silence item is skipped.
    assert metrics.corpus_wer(refs, hyps) == pytest.approx(1 / 10)


def calc_action(expression):
    from app.pipeline import tools
    return {"tool": "calculate", "args": {"expression": expression}, "result": tools.calculate(expression)}


def test_calculate_matches_on_result_not_expression_text():
    assert metrics.call_matches("calculate", {"result": 391}, calc_action("17*23"))
    assert metrics.call_matches("calculate", {"result": 22.5}, calc_action("90 / 4"))
    assert not metrics.call_matches("calculate", {"result": 12}, calc_action("15% of 80"))  # invalid expression


def test_reminder_and_weather_matching():
    reminder = {"tool": "set_reminder", "args": {"text": "Call Mom", "minutes": "20"}, "result": {}}
    assert metrics.call_matches("set_reminder", {"minutes": 20, "text": "mom"}, reminder)
    assert not metrics.call_matches("set_reminder", {"minutes": 30, "text": "mom"}, reminder)
    weather = {"tool": "get_weather", "args": {"city": "Saarbrucken"}, "result": {}}
    assert metrics.call_matches("get_weather", {"city": "saarbr"}, weather)


def test_score_item_categories():
    tool_item = {"category": "weather", "expected_tool": "get_weather", "expected_args": {"city": "berlin"}}
    right = [{"tool": "get_weather", "args": {"city": "Berlin"}, "result": {}}]
    wrong_city = [{"tool": "get_weather", "args": {"city": "Bern"}, "result": {}}]
    assert metrics.score_item(tool_item, "x", right, True)["task_success"]
    scored = metrics.score_item(tool_item, "x", wrong_city, True)
    assert scored["tool_selected"] and not scored["args_correct"] and not scored["task_success"]

    chit_chat = {"category": "no_tool", "expected_tool": None}
    assert metrics.score_item(chit_chat, "hi", [], True)["task_success"]
    assert metrics.score_item(chit_chat, "hi", right, True)["false_tool_call"]

    silence = {"category": "silence", "expected_tool": None}
    assert metrics.score_item(silence, "", [], False)["silence_rejected"]
    assert not metrics.score_item(silence, "Thank you for watching.", [], True)["task_success"]


def test_percentiles():
    assert metrics.percentiles([100, 200, 300, None]) == {"p50": 200.0, "p95": 290.0, "n": 3}
    assert metrics.percentiles([]) == {"p50": None, "p95": None, "n": 0}


def test_mu_law_roundtrip_is_close_but_lossy():
    signal = np.sin(np.linspace(0, 20 * np.pi, 8000)) * 0.5
    restored = audio.mu_law_roundtrip(signal)
    error = np.abs(restored - signal)
    assert error.max() < 0.02 and error.max() > 0


def test_telephony_produces_8khz_wav():
    tone = np.sin(2 * np.pi * 440 * np.arange(16000) / 16000) * 0.3
    out = audio.telephony(audio.to_wav_bytes(tone, 16000))
    with wave.open(io.BytesIO(out)) as w:
        assert w.getframerate() == 8000 and w.getnframes() == pytest.approx(8000, abs=200)
