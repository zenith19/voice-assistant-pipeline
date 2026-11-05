"""Runtime configuration, read from environment variables with local-dev defaults."""
import os
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"

# LLM provider:
#   "ollama"  - local Ollama server (default; fully offline, no account needed)
#   "openai"  - any other OpenAI-compatible endpoint (set LLM_BASE_URL / LLM_API_KEY)
#   "bedrock" - Amazon Bedrock Converse API; credentials come from the standard AWS chain
#               (AWS_PROFILE, env vars, ~/.aws/credentials)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()
_DEFAULT_MODELS = {"ollama": "qwen2.5:1.5b", "openai": "gpt-4o-mini", "bedrock": "eu.amazon.nova-lite-v1:0"}
if LLM_PROVIDER not in _DEFAULT_MODELS:
    raise ValueError(f"LLM_PROVIDER must be one of {sorted(_DEFAULT_MODELS)}, got {LLM_PROVIDER!r}")
LLM_MODEL = os.getenv("LLM_MODEL", _DEFAULT_MODELS[LLM_PROVIDER])
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://127.0.0.1:11434/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "ollama")
AWS_REGION = os.getenv("AWS_REGION", "eu-central-1")
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "512"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
MAX_TOOL_ROUNDS = int(os.getenv("MAX_TOOL_ROUNDS", "4"))
MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", "20"))

# ASR (faster-whisper)
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
# Segments whose no-speech probability exceeds this are treated as silence.
NO_SPEECH_THRESHOLD = float(os.getenv("NO_SPEECH_THRESHOLD", "0.6"))

# TTS (espeak-ng)
ESPEAK_BINARY = os.getenv("ESPEAK_BINARY", "espeak-ng")
