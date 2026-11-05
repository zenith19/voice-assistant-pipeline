"""Record the evaluation clips, one prompt at a time.

    python -m evaluation.record                 # record everything not recorded yet
    python -m evaluation.record --only c03,p02  # (re-)record specific items
    python -m evaluation.record --list-devices

Uses ALSA's `arecord` (16 kHz, mono, 16-bit) and `aplay` for playback. Clips are saved to
evaluation/audio/<id>.wav. Read the sentence naturally, as you would say it to an assistant;
small deviations from the script are fine - WER is computed against the script text, so re-record
if you said something noticeably different.
"""
import argparse
import json
import shutil
import signal
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np

EVAL_DIR = Path(__file__).resolve().parent
PROMPTS = EVAL_DIR / "prompts.jsonl"
AUDIO_DIR = EVAL_DIR / "audio"
RATE = 16000


def load_prompts() -> list[dict]:
    with PROMPTS.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def record_raw(device: str | None) -> bytes:
    cmd = ["arecord", "-q", "-t", "raw", "-f", "S16_LE", "-r", str(RATE), "-c", "1"]
    if device:
        cmd += ["-D", device]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    input("  recording... press Enter to stop ")
    proc.send_signal(signal.SIGINT)
    pcm, _ = proc.communicate()
    return pcm


def write_wav(path: Path, pcm: bytes) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(pcm)


def describe(pcm: bytes) -> str:
    samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768
    if samples.size == 0:
        return "empty recording"
    peak = float(np.abs(samples).max())
    note = "  <- very quiet, check the microphone" if peak < 0.05 else ("  <- clipping" if peak > 0.99 else "")
    return f"{samples.size / RATE:.1f} s, peak {peak:.2f}{note}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", help="comma-separated ids to (re-)record")
    parser.add_argument("--category", help="only record this category")
    parser.add_argument("--device", help="ALSA capture device, e.g. plughw:0,0 (default: system default)")
    parser.add_argument("--list-devices", action="store_true")
    args = parser.parse_args()

    if not shutil.which("arecord"):
        sys.exit("arecord not found (sudo apt install alsa-utils)")
    if args.list_devices:
        subprocess.run(["arecord", "-l"], check=False)
        return

    AUDIO_DIR.mkdir(exist_ok=True)
    items = load_prompts()
    if args.only:
        wanted = set(args.only.split(","))
        items = [i for i in items if i["id"] in wanted]
    else:
        items = [i for i in items if not (AUDIO_DIR / f"{i['id']}.wav").exists()]
    if args.category:
        items = [i for i in items if i["category"] == args.category]
    if not items:
        print("Nothing to record - all clips exist (use --only <ids> to re-record).")
        return

    print(f"{len(items)} clip(s) to record. Ctrl+C to stop at any time; finished clips are kept.\n")
    for n, item in enumerate(items, 1):
        print(f"[{n}/{len(items)}] {item['id']}  ({item['category']})")
        if item.get("instructions"):
            print(f"  >> {item['instructions']}")
        print(f'  SAY: "{item["text"]}"' if item["text"] else "  (no speech for this clip)")
        while True:
            input("  press Enter to start recording ")
            pcm = record_raw(args.device)
            print(f"  {describe(pcm)}")
            path = AUDIO_DIR / f"{item['id']}.wav"
            write_wav(path, pcm)
            choice = input("  [Enter] keep   [p] play back   [r] redo   [q] quit: ").strip().lower()
            while choice == "p":
                subprocess.run(["aplay", "-q", str(path)], check=False)
                choice = input("  [Enter] keep   [r] redo   [q] quit: ").strip().lower()
            if choice == "r":
                continue
            if choice == "q":
                print("Stopped. Run again to continue with the remaining clips.")
                return
            break
        print()
    print(f"Done. Clips are in {AUDIO_DIR}")


if __name__ == "__main__":
    main()
