"""The adapters, exercised through the real SDKs with the HTTP layer replaced.

These check both directions of each adapter: that what fleet sends is what the
SDK accepts and puts on the wire, and that what the SDK parses off the wire
comes back as the right messages. No request leaves the process.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from fleet import Agent, Message, Text, TextDelta, ToolCall, ToolEnd, ToolResult, user
from fleet.providers import resolve
from fleet.providers.anthropic import Messages
from fleet.providers.openai import ChatCompletions

try:  # the SDKs moved from httpx to httpx2; use whichever they are built on
    import httpx2 as httpx
except ImportError:  # pragma: no cover
    import httpx  # type: ignore[no-redef]


def get_weather(city: str) -> str:
    """Look up the weather.

    Args:
        city: The city to look up.
    """
    return f"Sunny in {city}"


class Wire:
    """A fake server: replies with canned HTTP responses and records each request."""

    def __init__(self, *replies: Any) -> None:
        self.replies = list(replies)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        reply = self.replies.pop(0)
        if isinstance(reply, dict):
            return httpx.Response(200, json=reply)
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=reply.encode())

    @property
    def bodies(self) -> list[dict[str, Any]]:
        return [json.loads(r.content) for r in self.requests]

    def client(self, cls: type, *, sync: bool = False) -> Any:
        transport = httpx.MockTransport(self)
        http = httpx.Client(transport=transport) if sync else httpx.AsyncClient(transport=transport)
        return cls(api_key="test", http_client=http, max_retries=0)


def test_resolve_picks_the_adapter_from_the_client():
    openai = pytest.importorskip("openai")
    anthropic = pytest.importorskip("anthropic")
    assert isinstance(resolve(openai.OpenAI(api_key="x")), ChatCompletions)
    assert isinstance(resolve(openai.AsyncOpenAI(api_key="x")), ChatCompletions)
    assert isinstance(resolve(anthropic.Anthropic(api_key="x")), Messages)
    assert isinstance(resolve(anthropic.AsyncAnthropic(api_key="x")), Messages)
    with pytest.raises(TypeError, match="Don't know how to talk to"):
        resolve(object())


# --- OpenAI Chat Completions -------------------------------------------------


def completion(message: dict[str, Any], finish: str, usage: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 0,
        "model": "gpt-test",
        "choices": [{"index": 0, "finish_reason": finish, "message": {"role": "assistant", **message}}],
        "usage": usage or {"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25},
    }


def function_call(id: str, name: str, arguments: str) -> dict[str, Any]:
    return {"id": id, "type": "function", "function": {"name": name, "arguments": arguments}}


def chunks(*deltas: dict[str, Any], finish: str, usage: dict[str, Any]) -> str:
    base = {"id": "chatcmpl-1", "object": "chat.completion.chunk", "created": 0, "model": "gpt-test"}
    events = [{**base, "choices": [{"index": 0, "delta": d, "finish_reason": None}]} for d in deltas]
    events.append({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": finish}]})
    events.append({**base, "choices": [], "usage": usage})
    return "".join(f"data: {json.dumps(e)}\n\n" for e in events) + "data: [DONE]\n\n"


WEATHER_CALL = function_call("call_1", "get_weather", '{"city": "Paris"}')


@pytest.mark.parametrize("sync", [False, True], ids=["async-client", "sync-client"])
async def test_openai_tool_loop(sync: bool):
    openai = pytest.importorskip("openai")
    wire = Wire(
        completion({"content": None, "tool_calls": [WEATHER_CALL]}, "tool_calls"),
        completion(
            {"content": "It is sunny in Paris."},
            "stop",
            {
                "prompt_tokens": 40,
                "completion_tokens": 8,
                "total_tokens": 48,
                "prompt_tokens_details": {"cached_tokens": 20},
            },
        ),
    )
    agent = Agent(
        client=wire.client(openai.OpenAI if sync else openai.AsyncOpenAI, sync=sync),
        model="gpt-test",
        instructions="Be helpful.",
        tools=[get_weather],
        params={"temperature": 0.2},
    )

    result = await agent.run("Weather in Paris?")

    assert result.output == "It is sunny in Paris."
    assert result.usage.requests == 2
    assert (result.usage.input_tokens, result.usage.output_tokens, result.usage.cached_tokens) == (60, 13, 20)
    first, second = wire.bodies
    assert first == {
        "model": "gpt-test",
        "temperature": 0.2,
        "messages": [{"role": "system", "content": "Be helpful."}, {"role": "user", "content": "Weather in Paris?"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Look up the weather.",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string", "description": "The city to look up."}},
                        "required": ["city"],
                        "additionalProperties": False,
                    },
                },
            }
        ],
    }
    assert second["messages"][2:] == [
        {"role": "assistant", "content": None, "tool_calls": [WEATHER_CALL]},
        {"role": "tool", "tool_call_id": "call_1", "content": "Sunny in Paris"},
    ]


@pytest.mark.parametrize("sync", [False, True], ids=["async-client", "sync-client"])
async def test_openai_streaming(sync: bool):
    openai = pytest.importorskip("openai")
    usage = {"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25}
    wire = Wire(
        chunks(
            {"role": "assistant", "content": "Let me check."},
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": ""},
                    }
                ]
            },
            {"tool_calls": [{"index": 0, "function": {"arguments": '{"city": '}}]},
            {"tool_calls": [{"index": 0, "function": {"arguments": '"Paris"}'}}]},
            finish="tool_calls",
            usage=usage,
        ),
        chunks({"content": "Sunny"}, {"content": " in Paris."}, finish="stop", usage=usage),
    )
    agent = Agent(
        client=wire.client(openai.OpenAI if sync else openai.AsyncOpenAI, sync=sync),
        model="gpt-test",
        tools=[get_weather],
    )

    events = [event async for event in agent.stream("Weather in Paris?")]

    assert [e.text for e in events if isinstance(e, TextDelta)] == ["Let me check.", "Sunny", " in Paris."]
    assert [e.result.content for e in events if isinstance(e, ToolEnd)] == ["Sunny in Paris"]
    result = events[-1].result
    assert result.output == "Sunny in Paris."
    assert result.messages[1].parts == [Text("Let me check."), ToolCall("call_1", "get_weather", {"city": "Paris"})]
    assert result.usage.input_tokens == 40
    assert wire.bodies[0]["stream"] is True and wire.bodies[0]["stream_options"] == {"include_usage": True}
    assert wire.bodies[1]["messages"][1] == {
        "role": "assistant",
        "content": "Let me check.",
        "tool_calls": [WEATHER_CALL],
    }


async def test_openai_malformed_arguments_are_answered_with_an_error_and_replayed_verbatim():
    openai = pytest.importorskip("openai")
    broken = function_call("call_1", "get_weather", '{"city": "Par')
    wire = Wire(
        completion({"content": None, "tool_calls": [broken]}, "tool_calls"),
        completion({"content": "Sorry."}, "stop"),
    )
    result = await Agent(client=wire.client(openai.AsyncOpenAI), model="gpt-test", tools=[get_weather]).run("Weather?")

    assert result.messages[1].tool_calls[0].arguments == '{"city": "Par'
    assert result.messages[2].parts[0].is_error
    assert wire.bodies[1]["messages"][1]["tool_calls"] == [broken]


# --- Anthropic Messages ------------------------------------------------------


def sse(content: list[dict[str, Any]], stop_reason: str, usage: dict[str, int]) -> str:
    """The event stream the Messages API would send for a reply with these content blocks."""
    start = {"id": "msg_1", "type": "message", "role": "assistant", "model": "claude-test", "content": []}
    events: list[dict[str, Any]] = [
        {
            "type": "message_start",
            "message": {**start, "stop_reason": None, "stop_sequence": None, "usage": {**usage, "output_tokens": 0}},
        }
    ]
    for index, block in enumerate(content):
        if block["type"] == "text":
            empty, deltas = (
                {"type": "text", "text": ""},
                [{"type": "text_delta", "text": piece} for piece in block["pieces"]],
            )
        elif block["type"] == "thinking":
            empty = {"type": "thinking", "thinking": "", "signature": ""}
            deltas = [{"type": "signature_delta", "signature": block["signature"]}]
        else:
            empty = {**block, "input": {}}
            deltas = [{"type": "input_json_delta", "partial_json": json.dumps(block["input"])}]
        events.append({"type": "content_block_start", "index": index, "content_block": empty})
        events += [{"type": "content_block_delta", "index": index, "delta": delta} for delta in deltas]
        events.append({"type": "content_block_stop", "index": index})
    events.append(
        {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": {"output_tokens": usage["output_tokens"]},
        }
    )
    events.append({"type": "message_stop"})
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events)


THINKING = {"type": "thinking", "signature": "sig-abc"}
WEATHER_USE = {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "Paris"}}
BROKEN_USE = {"type": "tool_use", "id": "toolu_2", "name": "get_weather", "input": {"town": "Paris"}}


@pytest.mark.parametrize("sync", [False, True], ids=["async-client", "sync-client"])
async def test_anthropic_tool_loop(sync: bool):
    anthropic = pytest.importorskip("anthropic")
    wire = Wire(
        sse(
            [THINKING, {"type": "text", "pieces": ["Let me ", "check."]}, WEATHER_USE, BROKEN_USE],
            "tool_use",
            {"input_tokens": 10, "cache_read_input_tokens": 100, "cache_creation_input_tokens": 5, "output_tokens": 7},
        ),
        sse(
            [{"type": "text", "pieces": ["Sunny", " in Paris."]}], "end_turn", {"input_tokens": 30, "output_tokens": 4}
        ),
    )
    agent = Agent(
        client=wire.client(anthropic.Anthropic if sync else anthropic.AsyncAnthropic, sync=sync),
        model="claude-test",
        instructions="Be helpful.",
        tools=[get_weather],
        params={"thinking": {"type": "adaptive"}},
    )

    events = [event async for event in agent.stream("Weather in Paris?")]

    assert [e.text for e in events if isinstance(e, TextDelta)] == ["Let me ", "check.", "Sunny", " in Paris."]
    result = events[-1].result
    assert result.output == "Sunny in Paris."
    assert (
        result.usage.requests,
        result.usage.input_tokens,
        result.usage.output_tokens,
        result.usage.cached_tokens,
    ) == (2, 145, 11, 100)

    first, second = wire.bodies
    assert first == {
        "model": "claude-test",
        "max_tokens": 16000,
        "stream": True,
        "system": "Be helpful.",
        "thinking": {"type": "adaptive"},
        "messages": [{"role": "user", "content": [{"type": "text", "text": "Weather in Paris?"}]}],
        "tools": [
            {
                "name": "get_weather",
                "description": "Look up the weather.",
                "input_schema": {
                    "type": "object",
                    "properties": {"city": {"type": "string", "description": "The city to look up."}},
                    "required": ["city"],
                    "additionalProperties": False,
                },
            }
        ],
    }
    # The assistant turn goes back exactly as it came, thinking block and signature included...
    assert second["messages"][1] == {
        "role": "assistant",
        "content": [
            {"type": "thinking", "thinking": "", "signature": "sig-abc"},
            {"type": "text", "text": "Let me check."},
            WEATHER_USE,
            BROKEN_USE,
        ],
    }
    # ...and every tool result is answered in one user message.
    results = second["messages"][2]["content"]
    assert second["messages"][2]["role"] == "user" and len(second["messages"]) == 3
    assert results[0] == {"type": "tool_result", "tool_use_id": "toolu_1", "content": "Sunny in Paris"}
    assert results[1]["is_error"] is True and results[1]["tool_use_id"] == "toolu_2"


async def test_anthropic_betas_use_the_beta_endpoint_and_params_override_defaults():
    anthropic = pytest.importorskip("anthropic")
    wire = Wire(sse([{"type": "text", "pieces": ["ok"]}], "end_turn", {"input_tokens": 1, "output_tokens": 1}))
    agent = Agent(
        client=wire.client(anthropic.AsyncAnthropic),
        model="claude-test",
        params={"betas": ["some-beta-2026-01-01"], "max_tokens": 64000},
    )

    assert (await agent.run("hi")).output == "ok"
    assert "some-beta-2026-01-01" in wire.requests[0].headers["anthropic-beta"]
    assert "betas" not in wire.bodies[0]
    assert wire.bodies[0]["max_tokens"] == 64000


async def test_anthropic_finish_reasons():
    anthropic = pytest.importorskip("anthropic")
    usage = {"input_tokens": 1, "output_tokens": 1}
    wire = Wire(
        sse([{"type": "text", "pieces": ["cut"]}], "max_tokens", usage),
        sse([{"type": "text", "pieces": ["no"]}], "refusal", usage),
    )
    agent = Agent(client=wire.client(anthropic.AsyncAnthropic), model="claude-test")
    assert (await agent.run("hi")).finish_reason == "length"
    assert (await agent.run("hi")).finish_reason == "refusal"


async def test_a_conversation_can_move_between_providers():
    """History produced by one provider is rebuilt from its parts for the other."""
    openai = pytest.importorskip("openai")
    anthropic = pytest.importorskip("anthropic")
    usage = {"input_tokens": 1, "output_tokens": 1}

    from_openai = [
        user("Weather in Paris?"),
        Message("assistant", [Text(""), ToolCall("call_1", "get_weather", {"city": "Paris"})]),
        Message("tool", [ToolResult("call_1", "Sunny in Paris")]),
    ]
    wire = Wire(sse([{"type": "text", "pieces": ["Sunny."]}], "end_turn", usage))
    claude = Agent(client=wire.client(anthropic.AsyncAnthropic), model="claude-test", tools=[get_weather])
    result = await claude.run("And tomorrow?", history=from_openai)
    assert wire.bodies[0]["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": "Weather in Paris?"}]},
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "call_1", "name": "get_weather", "input": {"city": "Paris"}}],
        },
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call_1", "content": "Sunny in Paris"}]},
        {"role": "user", "content": [{"type": "text", "text": "And tomorrow?"}]},
    ]

    wire = Wire(completion({"content": "Rain."}, "stop"))
    gpt = Agent(client=wire.client(openai.AsyncOpenAI), model="gpt-test", tools=[get_weather])
    await gpt.run("And the day after?", history=result.messages)
    assert wire.bodies[0]["messages"][-3:] == [
        {"role": "user", "content": "And tomorrow?"},
        {"role": "assistant", "content": "Sunny."},  # rebuilt from parts; the Anthropic payload is not sent to OpenAI
        {"role": "user", "content": "And the day after?"},
    ]
