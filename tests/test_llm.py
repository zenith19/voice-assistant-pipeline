import json

from app import config
from app.pipeline import llm
from tests.conftest import FakeMessage, tool_call


def tool_messages(client, request_index):
    return [m for m in client.requests[request_index]["messages"] if m["role"] == "tool"]


def test_plain_reply_is_stored_in_history(fake_client):
    client = fake_client(FakeMessage(content="Hi there!"))
    history = []
    turn = llm.run_turn(history, "hello", client=client)

    assert turn["reply"] == "Hi there!" and turn["actions"] == []
    assert len(turn["timings"]["llm_rounds_ms"]) == 1
    assert history == [{"role": "user", "content": "hello"},
                       {"role": "assistant", "content": "Hi there!"}]


def test_tool_call_result_is_fed_back_to_model(fake_client):
    client = fake_client(
        FakeMessage(tool_calls=[tool_call("calculate", '{"expression": "23 * 17"}')]),
        FakeMessage(content="It's 391."),
    )
    turn = llm.run_turn([], "what's 23 times 17", client=client)

    assert turn["reply"] == "It's 391."
    assert turn["actions"][0]["result"]["result"] == 391
    assert json.loads(tool_messages(client, 1)[0]["content"])["result"] == 391


def test_malformed_tool_json_is_returned_as_error_not_raised(fake_client):
    client = fake_client(
        FakeMessage(tool_calls=[tool_call("calculate", '{"expression": ')]),
        FakeMessage(content="Sorry, let me try again."),
    )
    turn = llm.run_turn([], "math", client=client)

    assert "not valid JSON" in turn["actions"][0]["result"]["error"]
    assert "error" in json.loads(tool_messages(client, 1)[0]["content"])


def test_wrong_tool_arguments_are_returned_as_error(fake_client):
    client = fake_client(
        FakeMessage(tool_calls=[tool_call("set_reminder", '{"what": "call mom"}')]),
        FakeMessage(content="When should I remind you?"),
    )
    turn = llm.run_turn([], "remind me", client=client)
    assert "invalid arguments" in turn["actions"][0]["result"]["error"]


def test_non_object_arguments_are_rejected(fake_client):
    client = fake_client(
        FakeMessage(tool_calls=[tool_call("calculate", '["1+1"]')]),
        FakeMessage(content="ok"),
    )
    turn = llm.run_turn([], "math", client=client)
    assert "JSON object" in turn["actions"][0]["result"]["error"]


def test_unknown_tool(fake_client):
    client = fake_client(
        FakeMessage(tool_calls=[tool_call("launch_rocket", "{}")]),
        FakeMessage(content="I can't do that."),
    )
    turn = llm.run_turn([], "launch", client=client)
    assert turn["actions"][0]["result"] == {"error": "unknown tool launch_rocket"}


def test_max_rounds_returns_fallback_and_still_records_history(fake_client):
    looping = [FakeMessage(tool_calls=[tool_call("get_current_time", "{}")])
               for _ in range(config.MAX_TOOL_ROUNDS)]
    client = fake_client(*looping)
    history = []
    turn = llm.run_turn(history, "what time is it", client=client)

    assert turn["reply"] == llm.FALLBACK_REPLY
    assert len(turn["actions"]) == config.MAX_TOOL_ROUNDS
    assert len(turn["timings"]["llm_rounds_ms"]) == config.MAX_TOOL_ROUNDS
    assert history[-1] == {"role": "assistant", "content": llm.FALLBACK_REPLY}


def test_history_is_trimmed(fake_client):
    history = [{"role": "user", "content": str(i)} for i in range(config.MAX_HISTORY_MESSAGES)]
    llm.run_turn(history, "hello", client=fake_client(FakeMessage(content="hi")))
    assert len(history) == config.MAX_HISTORY_MESSAGES
    assert history[-1]["content"] == "hi"
