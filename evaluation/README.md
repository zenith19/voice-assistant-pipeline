# Evaluation

Offline evaluation of the batch pipeline (ASR -> LLM with tools -> TTS) on a fixed set of
50 recorded utterances. It measures what a voice assistant needs to get right per turn:

- **ASR**: corpus-level word error rate after normalization (case, punctuation, contractions,
  fillers like "um", and number words vs digits) - see `metrics.normalize`.
- **Agent**: task success, tool selection, argument correctness, and false tool calls on
  chit-chat. Tool arguments are checked by meaning: `calculate` is scored on the computed
  result, numbers after conversion, strings by case-insensitive substring.
- **Silence handling**: silent/noise-only clips must produce no transcript and no LLM call.
- **Latency**: p50/p95 per stage (ASR, LLM, TTS) and per turn, after a warm-up turn that
  loads the models.

## Prompt set (`prompts.jsonl`)

| Category | Items | What it checks |
|---|---|---|
| time, calculate, weather, reminder | 27 | the right tool with the right arguments |
| no_tool | 8 | chit-chat answered without calling a tool |
| noisy | 6 | background music, speech, fan noise, quiet/distant speaker |
| pause | 6 | mid-sentence hesitations (also input for the turn-detection experiment) |
| silence | 3 | silence, desk taps, a cough |

Each item is an independent single-turn conversation.

## Recording

```bash
python -m evaluation.record              # walks through every prompt not yet recorded
python -m evaluation.record --only c03   # re-record one clip
```

Clips are saved as 16 kHz mono WAV in `evaluation/audio/`, which is git-ignored because the
recordings are a real person's voice.

## Running

```bash
python -m evaluation.run_eval                            # recorded audio, clean
python -m evaluation.run_eval --condition telephony      # simulated phone line (8 kHz, mu-law)
python -m evaluation.run_eval --mode text                # script text straight to the LLM
LLM_PROVIDER=bedrock python -m evaluation.run_eval       # another model, same clips
```

Comparing `--mode audio` with `--mode text` for the same model separates recognition errors
from model errors. Results are written to `results/` as JSON (every item, with transcript,
tool calls, reply, and timings) and a Markdown summary.

## Limitations

- 50 utterances from one speaker on one microphone: differences of a few items between
  configurations are within noise.
- The telephony condition approximates G.711 (8 kHz resampling + 8-bit mu-law); it does not
  simulate packet loss, jitter, or the 300 Hz high-pass of a real phone line.
- Latency is measured on a CPU-only laptop under memory pressure; absolute numbers will
  differ on other hardware, and the LLM time depends on the provider.
