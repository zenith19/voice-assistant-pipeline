"""LLM stage: transcript -> reply text, deciding whether to call tools along the way."""
import json
import logging
from functools import lru_cache
from typing import Any

from app import config

from . import tools

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a helpful voice assistant. Keep spoken replies short and conversational. "
    "You do not know the current date, time, weather, or any live data from your own "
    "knowledge - your training data is stale and any date/time/weather you might recall "
    "or guess is WRONG. For any question about the current time or date, math, weather, "
    "or reminders, you MUST call the matching tool (get_current_time, calculate, "
    "get_weather, set_reminder) and base your answer only on its result. "
    "Never invent or guess a date, time, or numeric answer yourself."
)

FALLBACK_REPLY = "Sorry, I couldn't finish that after a few tries."


@lru_cache(maxsize=1)
def _default_client():
    if config.LLM_PROVIDER == "bedrock":
        from .bedrock import BedrockChatClient

        return BedrockChatClient(region=config.AWS_REGION, max_tokens=config.LLM_MAX_TOKENS)

    from openai import OpenAI

    return OpenAI(base_url=config.LLM_BASE_URL, api_key=config.LLM_API_KEY)


def _execute_tool_call(name: str, raw_arguments: str | None) -> tuple[Any, dict]:
    """Run one tool call. Never raises: malformed or failing calls become an error result
    that is sent back to the model, so it can correct itself on the next round."""
    try:
        args = json.loads(raw_arguments or "{}")
    except json.JSONDecodeError as exc:
        return raw_arguments, {"error": f"arguments are not valid JSON: {exc}"}
    if not isinstance(args, dict):
        return args, {"error": "arguments must be a JSON object"}

    handler = tools.DISPATCH.get(name)
    if handler is None:
        return args, {"error": f"unknown tool {name}"}
    try:
        return args, handler(**args)
    except TypeError as exc:  # wrong / missing keyword arguments
        return args, {"error": f"invalid arguments for {name}: {exc}"}
    except Exception as exc:
        logger.exception("tool %s failed", name)
        return args, {"error": f"{name} failed: {exc}"}


def _remember(history: list[dict], user_text: str, reply_text: str) -> None:
    history.append({"role": "user", "content": user_text})
    history.append({"role": "assistant", "content": reply_text})
    del history[:-config.MAX_HISTORY_MESSAGES]


def run_turn(history: list[dict], user_text: str, client=None) -> dict:
    """Run one voice turn. Mutates `history` in place with the new user/assistant messages."""
    client = client or _default_client()
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, *history, {"role": "user", "content": user_text}]
    actions = []

    for _ in range(config.MAX_TOOL_ROUNDS):
        response = client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=messages,
            tools=tools.TOOLS,
            temperature=config.LLM_TEMPERATURE,
        )
        message = response.choices[0].message

        if not message.tool_calls:
            reply_text = message.content or ""
            _remember(history, user_text, reply_text)
            return {"reply": reply_text, "actions": actions}

        messages.append(message.model_dump(exclude_none=True))
        for call in message.tool_calls:
            args, result = _execute_tool_call(call.function.name, call.function.arguments)
            actions.append({"tool": call.function.name, "args": args, "result": result})
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": json.dumps(result),
            })

    # Still record the turn, so the next turn keeps the conversational context.
    _remember(history, user_text, FALLBACK_REPLY)
    return {"reply": FALLBACK_REPLY, "actions": actions}
