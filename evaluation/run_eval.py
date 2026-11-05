"""Offline evaluation of the batch voice pipeline on the recorded prompt set.

    python -m evaluation.run_eval                          # recorded audio, clean
    python -m evaluation.run_eval --condition telephony    # 8 kHz mu-law simulated phone line
    python -m evaluation.run_eval --mode text              # reference text -> LLM (no ASR)

Running `--mode audio` and `--mode text` with the same model separates ASR errors from LLM
errors: an item that fails from audio but passes from text failed because of recognition.
The LLM provider/model come from the usual env vars (LLM_PROVIDER, LLM_MODEL, ...).
"""
import argparse
import json
import platform
import subprocess
import sys
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

from app import config
from app.pipeline import asr, turn
from evaluation import audio as audio_conditions
from evaluation import metrics

EVAL_DIR = Path(__file__).resolve().parent
PROMPTS = EVAL_DIR / "prompts.jsonl"
AUDIO_DIR = EVAL_DIR / "audio"
RESULTS_DIR = EVAL_DIR / "results"
LATENCY_KEYS = ("asr_ms", "llm_ms", "tts_ms", "total_ms")


def load_prompts(path: Path = PROMPTS) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              cwd=EVAL_DIR, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _rate(flags: list[bool]) -> dict:
    return {"rate": round(sum(flags) / len(flags), 3) if flags else None, "k": sum(flags), "n": len(flags)}


def summarize(rows: list[dict], mode: str) -> dict:
    ok = [r for r in rows if "error" not in r]
    speech = [r for r in ok if r["category"] != "silence"]
    tool_items = [r for r in speech if r["expected_tool"]]
    no_tool_items = [r for r in speech if not r["expected_tool"]]
    silence = [r for r in ok if r["category"] == "silence"]

    by_category = defaultdict(list)
    for r in ok:
        by_category[r["category"]].append(r["task_success"])

    summary = {
        "items": len(rows),
        "errors": len(rows) - len(ok),
        "task_success": _rate([r["task_success"] for r in ok]),
        "task_success_by_category": {c: _rate(v) for c, v in sorted(by_category.items())},
        "tool_selection": _rate([r["tool_selected"] for r in tool_items]),
        "args_correct": _rate([r["args_correct"] for r in tool_items]),
        "false_tool_calls_on_chit_chat": _rate([r["false_tool_call"] for r in no_tool_items]),
        "latency_ms": {k: metrics.percentiles([r["timings"].get(k) for r in ok if r["llm_called"] or k == "asr_ms"])
                       for k in LATENCY_KEYS},
        "mean_llm_rounds": round(sum(len(r["timings"]["llm_rounds_ms"]) for r in speech) / len(speech), 2)
        if speech else None,
    }
    if mode == "audio":
        summary["wer"] = metrics.corpus_wer([r["reference"] for r in speech], [r["transcript"] for r in speech])
        summary["silence_rejected"] = _rate([r["silence_rejected"] for r in silence])
    return summary


def _fmt_rate(r: dict) -> str:
    return "n/a" if r["rate"] is None else f"{r['rate']:.0%} ({r['k']}/{r['n']})"


def render_markdown(result: dict) -> str:
    meta, s = result["meta"], result["summary"]
    lines = [
        f"## Eval run {meta['timestamp']}",
        "",
        f"- Mode: `{meta['mode']}`, condition: `{meta['condition']}`, commit `{meta['commit']}`",
        (f"- LLM: `{meta['provider']}` / `{meta['model']}`; ASR: faster-whisper `{meta['whisper_model']}` "
         f"({meta['whisper_compute_type']}, {meta['whisper_device']})"),
        f"- Items: {s['items']} ({s['errors']} errors)",
        "",
        "| Metric | Value |",
        "|---|---|",
    ]
    if "wer" in s:
        lines.append(f"| WER (normalized) | {'n/a' if s['wer'] is None else f'{s['wer']:.1%}'} |")
    lines += [
        f"| Task success | {_fmt_rate(s['task_success'])} |",
        f"| Tool selection (tool items) | {_fmt_rate(s['tool_selection'])} |",
        f"| Tool arguments correct | {_fmt_rate(s['args_correct'])} |",
        f"| False tool calls on chit-chat | {_fmt_rate(s['false_tool_calls_on_chit_chat'])} |",
    ]
    if "silence_rejected" in s:
        lines.append(f"| Silence rejected | {_fmt_rate(s['silence_rejected'])} |")
    lines += ["", "| Category | Task success |", "|---|---|"]
    lines += [f"| {c} | {_fmt_rate(r)} |" for c, r in s["task_success_by_category"].items()]
    lines += ["", "| Stage | p50 ms | p95 ms | n |", "|---|---|---|---|"]
    lines += [f"| {k.removesuffix('_ms')} | {v['p50']} | {v['p95']} | {v['n']} |" for k, v in s["latency_ms"].items()]
    lines += ["", f"Mean LLM rounds per turn: {s['mean_llm_rounds']}", ""]

    failures = [r for r in result["rows"] if not r.get("task_success")]
    if failures:
        lines += ["### Failures", "", "| id | transcript / input | tools called | reply |", "|---|---|---|---|"]
        for r in failures:
            detail = r.get("error") or r["reply"]
            heard = r["transcript"] if meta["mode"] == "audio" else r["reference"]
            lines.append(f"| {r['id']} | {heard!r} | {r.get('tools_called')} | {detail[:80]!r} |".replace("\n", " "))
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> dict:
    items = load_prompts()
    if args.ids:
        items = [i for i in items if i["id"] in set(args.ids.split(","))]
    if args.categories:
        items = [i for i in items if i["category"] in set(args.categories.split(","))]
    if args.mode == "text":
        items = [i for i in items if i["category"] != "silence"]  # nothing to send without audio
    else:
        missing = [i["id"] for i in items if not (args.audio_dir / f"{i['id']}.wav").exists()]
        if missing:
            print(f"warning: no recording for {len(missing)} item(s), skipping: {', '.join(missing)}", file=sys.stderr)
        items = [i for i in items if i["id"] not in missing]
    if args.limit:
        items = items[: args.limit]
    if not items:
        sys.exit("nothing to evaluate (record clips with `python -m evaluation.record` first, or use --mode text)")

    # Warm-up (excluded from results): load Whisper and make sure the LLM is loaded/reachable.
    print("warming up...", file=sys.stderr)
    if args.mode == "audio":
        asr.preload()
    turn.respond([], "Hello.", synthesize=False)

    rows = []
    for n, item in enumerate(items, 1):
        row = {"id": item["id"], "category": item["category"], "reference": item["text"],
               "expected_tool": item["expected_tool"], "expected_args": item.get("expected_args", {})}
        try:
            history: list[dict] = []  # every item is an independent single-turn conversation
            if args.mode == "audio":
                clip = (args.audio_dir / f"{item['id']}.wav").read_bytes()
                if args.condition == "telephony":
                    clip = audio_conditions.telephony(clip)
                out = turn.run_voice_turn(history, clip, synthesize=not args.no_tts)
            else:
                started = time.perf_counter()
                out = turn.respond(history, item["text"], synthesize=not args.no_tts)
                out["timings"]["total_ms"] = round((time.perf_counter() - started) * 1000, 1)
            row |= {"transcript": out["transcript"], "reply": out["reply"], "actions": out["actions"],
                    "llm_called": out["llm_called"], "timings": out["timings"]}
            row |= metrics.score_item(item, out["transcript"], out["actions"], out["llm_called"])
        except Exception as exc:  # noqa: BLE001 - record and continue; one outage shouldn't lose the run
            row |= {"error": f"{type(exc).__name__}: {exc}", "task_success": False}
        status = "ERROR" if "error" in row else ("ok" if row["task_success"] else "FAIL")
        print(f"[{n}/{len(items)}] {item['id']:4} {status:5} {row.get('transcript', row['reference'])!r}",
              file=sys.stderr)
        rows.append(row)

    meta = {
        "timestamp": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "commit": _git_commit(), "mode": args.mode, "condition": args.condition if args.mode == "audio" else "n/a",
        "provider": config.LLM_PROVIDER, "model": config.LLM_MODEL, "temperature": config.LLM_TEMPERATURE,
        "whisper_model": config.WHISPER_MODEL, "whisper_device": config.WHISPER_DEVICE,
        "whisper_compute_type": config.WHISPER_COMPUTE_TYPE, "whisper_language": config.WHISPER_LANGUAGE,
        "tts": not args.no_tts, "machine": f"{platform.system()} {platform.machine()}",
    }
    return {"meta": meta, "summary": summarize(rows, args.mode), "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["audio", "text"], default="audio")
    parser.add_argument("--condition", choices=["clean", "telephony"], default="clean")
    parser.add_argument("--ids", help="comma-separated item ids")
    parser.add_argument("--categories", help="comma-separated categories")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--audio-dir", type=Path, default=AUDIO_DIR, help="where <id>.wav clips live")
    parser.add_argument("--no-tts", action="store_true", help="skip TTS (no tts_ms)")
    parser.add_argument("--tag", default="", help="suffix for the result file name")
    args = parser.parse_args()

    result = run(args)
    RESULTS_DIR.mkdir(exist_ok=True)
    model = config.LLM_MODEL.replace(":", "-").replace("/", "-")
    condition = args.condition if args.mode == "audio" else "text"
    stem = f"{result['meta']['timestamp'][:19].replace(':', '')}_{config.LLM_PROVIDER}_{model}_{condition}"
    stem += f"_{args.tag}" if args.tag else ""
    (RESULTS_DIR / f"{stem}.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    markdown = render_markdown(result)
    (RESULTS_DIR / f"{stem}.md").write_text(markdown)
    print(markdown)
    print(f"saved {RESULTS_DIR / stem}.{{json,md}}", file=sys.stderr)


if __name__ == "__main__":
    main()
