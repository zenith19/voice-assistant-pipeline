"""LLM stage: transcript -> reply text, deciding whether to call tools along the way."""
import json

from openai import OpenAI

from . import tools

SYSTEM_PROMPT = (
    "You are a helpful voice assistant. Keep spoken replies short and conversational. "
    "You do not know the current date, time, weather, or any live data from your own "
    "knowledge - your training data is stale and any date/time/weather you might recall "
    "or guess is WRONG. For any question about the current time or date, math, weather, "
    "or reminders, you MUST call the matching tool (get_current_time, calculate, "
    "get_weather, set_reminder) and base your answer only on its result. "
    "Never invent or guess a date, time, or numeric answer yourself."
)

MODEL = "qwen2.5:1.5b"
MAX_TOOL_ROUNDS = 4
MAX_HISTORY_MESSAGES = 20

client = OpenAI(base_url="http://127.0.0.1:11434/v1", api_key="ollama")


def run_turn(history: list[dict], user_text: str) -> dict:
    """Run one voice turn. Mutates `history` in place with the new user/assistant messages."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, *history, {"role": "user", "content": user_text}]
    actions = []

    for _ in range(MAX_TOOL_ROUNDS):
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools.TOOLS,
            temperature=0.1,
        )
        message = response.choices[0].message

        if not message.tool_calls:
            reply_text = message.content or ""
            history.append({"role": "user", "content": user_text})
            history.append({"role": "assistant", "content": reply_text})
            del history[:-MAX_HISTORY_MESSAGES]
            return {"reply": reply_text, "actions": actions}

        messages.append(message.model_dump(exclude_none=True))
        for call in message.tool_calls:
            args = json.loads(call.function.arguments or "{}")
            handler = tools.DISPATCH.get(call.function.name)
            result = handler(**args) if handler else {"error": f"unknown tool {call.function.name}"}
            actions.append({"tool": call.function.name, "args": args, "result": result})
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": json.dumps(result),
            })

    return {"reply": "Sorry, I couldn't finish that after a few tries.", "actions": actions}
