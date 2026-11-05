import json

from app.pipeline import bedrock, llm, tools


class FakeRuntime:
    """Stands in for boto3's bedrock-runtime client; replays scripted Converse responses."""

    def __init__(self, *responses):
        self._responses = list(responses)
        self.calls = []

    def converse(self, **kwargs):
        self.calls.append(kwargs)
        return self._responses.pop(0)


def text_response(text):
    return {"output": {"message": {"role": "assistant", "content": [{"text": text}]}},
            "stopReason": "end_turn", "usage": {"inputTokens": 10, "outputTokens": 5}}


def tool_use_response(name, tool_input, tool_use_id="tu_1", preamble=None):
    content = [{"text": preamble}] if preamble else []
    content.append({"toolUse": {"toolUseId": tool_use_id, "name": name, "input": tool_input}})
    return {"output": {"message": {"role": "assistant", "content": content}}, "stopReason": "tool_use"}


def test_messages_are_converted_to_converse_format():
    system, messages = bedrock.to_converse_messages([
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "what time is it and what's 2+2"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "a", "type": "function", "function": {"name": "get_current_time", "arguments": "{}"}},
            {"id": "b", "type": "function", "function": {"name": "calculate", "arguments": '{"expression": "2+2"}'}},
        ]},
        {"role": "tool", "tool_call_id": "a", "content": '{"current_time": "12:00"}'},
        {"role": "tool", "tool_call_id": "b", "content": '{"result": 4}'},
    ])

    assert system == [{"text": "be brief"}]
    assert [m["role"] for m in messages] == ["user", "assistant", "user"]
    assert messages[1]["content"][1]["toolUse"] == {"toolUseId": "b", "name": "calculate", "input": {"expression": "2+2"}}
    # Both tool results are merged into one user message (Converse requires alternating roles).
    assert [b["toolResult"]["toolUseId"] for b in messages[2]["content"]] == ["a", "b"]


def test_consecutive_user_messages_are_merged():
    _system, messages = bedrock.to_converse_messages([
        {"role": "user", "content": "one"}, {"role": "user", "content": "two"},
    ])
    assert messages == [{"role": "user", "content": [{"text": "one"}, {"text": "two"}]}]


def test_tool_schemas_become_tool_specs():
    specs = bedrock.to_converse_tools(tools.TOOLS)["tools"]
    assert {s["toolSpec"]["name"] for s in specs} == set(tools.DISPATCH)
    calc = next(s["toolSpec"] for s in specs if s["toolSpec"]["name"] == "calculate")
    assert calc["inputSchema"]["json"]["required"] == ["expression"]


def test_tool_use_response_maps_to_openai_style_tool_calls():
    response = bedrock.from_converse_response(
        tool_use_response("calculate", {"expression": "1+1"}, preamble="Let me check."))
    message = response.choices[0].message
    assert message.content == "Let me check."
    assert message.tool_calls[0].function.name == "calculate"
    assert json.loads(message.tool_calls[0].function.arguments) == {"expression": "1+1"}


def test_run_turn_end_to_end_through_adapter():
    runtime = FakeRuntime(
        tool_use_response("calculate", {"expression": "23 * 17"}),
        text_response("That's 391."),
    )
    client = bedrock.BedrockChatClient(runtime_client=runtime, max_tokens=256)
    history = []
    turn = llm.run_turn(history, "what's 23 times 17", client=client)

    assert turn["reply"] == "That's 391."
    assert turn["actions"][0]["result"]["result"] == 391

    first, second = runtime.calls
    assert first["system"][0]["text"] == llm.SYSTEM_PROMPT
    assert first["inferenceConfig"]["maxTokens"] == 256
    assert "toolConfig" in first and "toolConfig" in second
    tool_result = second["messages"][-1]["content"][0]["toolResult"]
    assert tool_result["toolUseId"] == "tu_1"
    assert json.loads(tool_result["content"][0]["text"])["result"] == 391
    assert history[-1] == {"role": "assistant", "content": "That's 391."}
