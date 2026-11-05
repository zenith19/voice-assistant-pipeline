"""Amazon Bedrock adapter with the same `chat.completions.create(...)` shape as the OpenAI client.

`llm.run_turn` speaks OpenAI-style chat messages (system/user/assistant/tool, `tool_calls`).
This adapter translates them to the Bedrock Converse API and translates the response back, so
the agent loop, the tools, and their tests are provider-agnostic.
"""
import json
from types import SimpleNamespace
from typing import Any


class BedrockMessage:
    """Mimics the parts of `openai...ChatCompletionMessage` that `run_turn` uses."""

    def __init__(self, content: str | None, tool_calls: list | None):
        self.content = content
        self.tool_calls = tool_calls or None

    def model_dump(self, exclude_none: bool = False) -> dict:
        data: dict[str, Any] = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            data["tool_calls"] = [
                {"id": c.id, "type": "function",
                 "function": {"name": c.function.name, "arguments": c.function.arguments}}
                for c in self.tool_calls
            ]
        return {k: v for k, v in data.items() if not (exclude_none and v is None)}


def to_converse_tools(tools: list[dict]) -> dict:
    return {"tools": [
        {"toolSpec": {
            "name": t["function"]["name"],
            "description": t["function"].get("description", ""),
            "inputSchema": {"json": t["function"].get("parameters", {"type": "object", "properties": {}})},
        }}
        for t in tools
    ]}


def _parse_arguments(raw: str | None) -> dict:
    try:
        args = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {}
    return args if isinstance(args, dict) else {}


def to_converse_messages(messages: list[dict]) -> tuple[list[dict], list[dict]]:
    """Split OpenAI-style messages into Converse (`system`, `messages`).

    Converse differences handled here:
    - system prompts are a separate parameter;
    - tool results are `toolResult` blocks inside a *user* message;
    - consecutive messages with the same role must be merged into one.
    """
    system: list[dict] = []
    converse: list[dict] = []

    def append(role: str, blocks: list[dict]) -> None:
        if converse and converse[-1]["role"] == role:
            converse[-1]["content"].extend(blocks)
        else:
            converse.append({"role": role, "content": blocks})

    for m in messages:
        role = m["role"]
        if role == "system":
            system.append({"text": m["content"]})
        elif role == "user":
            append("user", [{"text": m["content"]}])
        elif role == "assistant":
            blocks = [{"text": m["content"]}] if m.get("content") else []
            for call in m.get("tool_calls") or []:
                blocks.append({"toolUse": {
                    "toolUseId": call["id"],
                    "name": call["function"]["name"],
                    "input": _parse_arguments(call["function"].get("arguments")),
                }})
            if blocks:
                append("assistant", blocks)
        elif role == "tool":
            append("user", [{"toolResult": {
                "toolUseId": m["tool_call_id"],
                "content": [{"text": m["content"]}],
            }}])
        else:
            raise ValueError(f"unsupported message role: {role}")
    return system, converse


def from_converse_response(response: dict) -> SimpleNamespace:
    blocks = response["output"]["message"]["content"]
    text = "".join(b["text"] for b in blocks if "text" in b).strip() or None
    tool_calls = [
        SimpleNamespace(
            id=b["toolUse"]["toolUseId"],
            function=SimpleNamespace(name=b["toolUse"]["name"], arguments=json.dumps(b["toolUse"].get("input", {}))),
        )
        for b in blocks if "toolUse" in b
    ]
    usage = response.get("usage", {})
    return SimpleNamespace(
        choices=[SimpleNamespace(message=BedrockMessage(text, tool_calls), finish_reason=response.get("stopReason"))],
        usage=SimpleNamespace(prompt_tokens=usage.get("inputTokens"), completion_tokens=usage.get("outputTokens")),
    )


class BedrockChatClient:
    """`client.chat.completions.create(model=..., messages=..., tools=..., temperature=...)` on Bedrock."""

    def __init__(self, runtime_client=None, region: str = "eu-central-1", max_tokens: int = 512):
        if runtime_client is None:
            import boto3
            from botocore.config import Config

            # Short timeouts: a voice turn that waits 60 s for the model is already a failed turn.
            runtime_client = boto3.client("bedrock-runtime", region_name=region, config=Config(
                connect_timeout=3, read_timeout=20, retries={"mode": "standard", "max_attempts": 3},
            ))
        self._runtime = runtime_client
        self._max_tokens = max_tokens
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, *, model: str, messages: list[dict], tools: list[dict] | None = None,
                temperature: float | None = None) -> SimpleNamespace:
        system, converse_messages = to_converse_messages(messages)
        inference: dict[str, Any] = {"maxTokens": self._max_tokens}
        if temperature is not None:
            inference["temperature"] = temperature
        kwargs: dict[str, Any] = {"modelId": model, "messages": converse_messages, "inferenceConfig": inference}
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["toolConfig"] = to_converse_tools(tools)
        return from_converse_response(self._runtime.converse(**kwargs))
