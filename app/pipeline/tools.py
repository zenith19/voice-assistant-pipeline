"""Demo tools the assistant can call: implementations + their function-calling schemas."""
import ast
import operator
from datetime import datetime
from typing import Any

_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

_reminders: list[dict[str, Any]] = []

_MOCK_WEATHER = {
    "munich": "18C and cloudy",
    "berlin": "21C and sunny",
    "new york": "25C and humid",
}


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPERATORS:
        return _ALLOWED_OPERATORS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPERATORS:
        return _ALLOWED_OPERATORS[type(node.op)](_eval_node(node.operand))
    raise ValueError(f"unsupported expression near: {ast.dump(node)}")


def calculate(expression: str) -> dict:
    """Safely evaluate basic arithmetic (no eval() - only whitelisted AST nodes)."""
    try:
        result = _eval_node(ast.parse(expression, mode="eval").body)
        return {"expression": expression, "result": result}
    except Exception as exc:
        return {"expression": expression, "error": str(exc)}


def get_current_time() -> dict:
    return {"current_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}


def set_reminder(text: str, minutes: int) -> dict:
    _reminders.append({"text": text, "minutes": minutes, "created_at": datetime.now().isoformat()})
    return {"status": "reminder set", "text": text, "minutes": minutes, "total_reminders": len(_reminders)}


def get_weather(city: str) -> dict:
    forecast = _MOCK_WEATHER.get(city.strip().lower(), "22C and partly cloudy")
    return {"city": city, "forecast": forecast, "note": "demo data, not a real weather API"}


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current date and time.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a basic arithmetic expression, e.g. '12 * (3 + 4)'.",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_reminder",
            "description": "Set a reminder for N minutes from now.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "What to be reminded about"},
                    "minutes": {"type": "integer", "description": "Minutes from now"},
                },
                "required": ["text", "minutes"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get a (mock/demo) weather forecast for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    },
]

DISPATCH = {
    "get_current_time": lambda **kwargs: get_current_time(),
    "calculate": lambda **kwargs: calculate(**kwargs),
    "set_reminder": lambda **kwargs: set_reminder(**kwargs),
    "get_weather": lambda **kwargs: get_weather(**kwargs),
}
