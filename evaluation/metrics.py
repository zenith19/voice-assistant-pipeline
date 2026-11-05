"""Scoring helpers for the offline evaluation: text normalization for WER, tool-call matching,
and latency percentiles."""
import math
import re

import jiwer
import numpy as np

_CONTRACTIONS = {
    "what's": "what is", "that's": "that is", "it's": "it is", "how's": "how is", "where's": "where is",
    "there's": "there is", "let's": "let us", "i'm": "i am", "you're": "you are", "we're": "we are",
    "don't": "do not", "doesn't": "does not", "can't": "can not", "cannot": "can not", "won't": "will not",
    "i'll": "i will", "you'll": "you will", "i'd": "i would", "i've": "i have", "isn't": "is not",
}
_FILLERS = {"um", "uh", "erm", "er", "hmm", "mm", "ah"}

_UNITS = {w: i for i, w in enumerate([
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
    "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"])}
_TENS = {w: 10 * i for i, w in enumerate(
    ["twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"], start=2)}
_SCALES = {"hundred": 100, "thousand": 1000}


def _words_to_digits(tokens: list[str]) -> list[str]:
    """Collapse runs of English number words into digits ("twenty three" -> "23"), so that a
    reference and a transcript that differ only in how numbers are written score the same."""
    out: list[str] = []
    total = current = 0
    in_number = False

    def flush():
        nonlocal total, current, in_number
        if in_number:
            out.append(str(total + current))
        total = current = 0
        in_number = False

    for tok in tokens:
        if tok in _UNITS:
            current += _UNITS[tok]
        elif tok in _TENS:
            current += _TENS[tok]
        elif tok in _SCALES and in_number:
            scale = _SCALES[tok]
            if scale == 100:
                current = max(current, 1) * 100
            else:
                total += max(current, 1) * scale
                current = 0
        elif tok == "and" and in_number:
            continue  # "one hundred and five"
        else:
            flush()
            out.append(tok)
            continue
        in_number = True
    flush()
    return out


def normalize(text: str) -> str:
    """Lowercase, expand contractions, drop fillers and punctuation, unify numbers.

    A deliberately small normalizer (not Whisper's full EnglishTextNormalizer); it covers the
    formatting differences that occur in this eval set, so WER reflects recognition errors.
    """
    text = text.lower().replace("’", "'").replace("%", " percent").replace("&", " and ")
    text = re.sub(r"(?<=\d),(?=\d{3})", "", text)       # 1,000 -> 1000
    text = re.sub(r"(?<=\d):(?=\d)", " ", text)          # 7:30 -> 7 30
    text = re.sub(r"(?<=[a-z])-(?=[a-z])", " ", text)    # twenty-three -> twenty three
    tokens = []
    for tok in text.split():
        tok = re.sub(r"^[^\w']+|[^\w']+$", "", tok)       # strip surrounding punctuation
        tok = re.sub(r"(?<!\d)\.|\.(?!\d)", "", tok)      # keep decimal points only
        if not tok:
            continue
        tokens.extend(_CONTRACTIONS.get(tok, tok).split())
    tokens = [t.replace("'", "") for t in tokens if t not in _FILLERS]
    return " ".join(_words_to_digits(tokens))


def corpus_wer(references: list[str], hypotheses: list[str]) -> float | None:
    """Corpus-level WER (total edits / total reference words) over normalized text."""
    pairs = [(normalize(r), normalize(h)) for r, h in zip(references, hypotheses, strict=True)]
    pairs = [(r, h) for r, h in pairs if r]
    if not pairs:
        return None
    refs, hyps = zip(*pairs, strict=True)
    return jiwer.wer(list(refs), list(hyps))


def _number(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def call_matches(tool: str, expected_args: dict, action: dict) -> bool:
    """Does one executed tool call (`{"tool", "args", "result"}`) satisfy the expected arguments?

    - calculate: compares the *computed result*, so "23*17" and "17 * 23" both count;
    - numbers: equal after float conversion ("20" == 20);
    - strings: case-insensitive substring ("saarbr" matches "Saarbrücken" and "Saarbrucken").
    """
    if action.get("tool") != tool:
        return False
    args = action.get("args") if isinstance(action.get("args"), dict) else {}
    for key, expected in expected_args.items():
        if tool == "calculate" and key == "result":
            actual = _number((action.get("result") or {}).get("result"))
            if actual is None or not math.isclose(actual, float(expected), rel_tol=1e-6, abs_tol=1e-9):
                return False
        elif isinstance(expected, (int, float)):
            actual = _number(args.get(key))
            if actual is None or not math.isclose(actual, float(expected)):
                return False
        elif str(expected).lower() not in str(args.get(key, "")).lower():
            return False
    return True


def score_item(item: dict, transcript: str, actions: list[dict], llm_called: bool) -> dict:
    """Per-utterance outcome. `task_success` is the single headline flag:
    tool items need the right tool with the right arguments, no-tool items need no tool call,
    silence items need an empty transcript and no LLM call."""
    expected_tool = item.get("expected_tool")
    called = [a.get("tool") for a in actions]

    if item["category"] == "silence":
        rejected = not normalize(transcript) and not llm_called
        return {"task_success": rejected, "silence_rejected": rejected, "tools_called": called}

    if expected_tool is None:
        return {"task_success": not called, "false_tool_call": bool(called), "tools_called": called}

    tool_selected = expected_tool in called
    args_correct = any(call_matches(expected_tool, item.get("expected_args", {}), a) for a in actions)
    return {
        "task_success": tool_selected and args_correct,
        "tool_selected": tool_selected,
        "args_correct": args_correct,
        "extra_tools": [t for t in called if t != expected_tool],
        "tools_called": called,
    }


def percentiles(values: list[float], qs=(50, 95)) -> dict:
    values = [v for v in values if v is not None]
    if not values:
        return {f"p{q}": None for q in qs} | {"n": 0}
    return {f"p{q}": round(float(np.percentile(values, q)), 1) for q in qs} | {"n": len(values)}
