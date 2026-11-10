from __future__ import annotations

import asyncio
import dataclasses
import json
import time
from dataclasses import dataclass

import pytest
from conftest import Script, call, calls, say
from pydantic import BaseModel

from fleet import (
    Agent,
    Context,
    Done,
    FleetError,
    Handoff,
    MaxTurnsExceeded,
    Message,
    TextDelta,
    ToolEnd,
    ToolStart,
    user,
)


def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


async def test_plain_answer():
    script = Script(say("Hello!"))
    agent = Agent(name="greeter", instructions="Be brief.", client=script, model="m", params={"temperature": 0})

    result = await agent.run("Hi")

    assert result.output == "Hello!"
    assert result.agent is agent
    assert result.finish_reason == "stop"
    assert [m.role for m in result.messages] == ["user", "assistant"]
    request = script.requests[0]
    assert (request.model, request.instructions, request.params) == ("m", "Be brief.", {"temperature": 0})


async def test_tool_loop_feeds_results_back():
    script = Script(call("add", {"a": 2, "b": 3}), say("5"))
    agent = Agent(client=script, model="m", tools=[add])

    result = await agent.run("2+3?")

    assert result.output == "5"
    assert [m.role for m in result.messages] == ["user", "assistant", "tool", "assistant"]
    tool_result = result.messages[2].parts[0]
    assert (tool_result.id, tool_result.content, tool_result.is_error) == ("call_0", "5", False)
    assert script.requests[1].messages[-1].role == "tool"  # the model saw the result
    assert (result.usage.requests, result.usage.input_tokens, result.usage.output_tokens) == (2, 20, 10)


async def test_tool_calls_in_one_turn_run_concurrently_and_stay_in_order():
    def slow(label: str) -> str:
        time.sleep(0.2)  # a blocking tool: it must not hold up the others
        return label

    script = Script(calls(("slow", {"label": "a"}), ("slow", {"label": "b"}), ("slow", {"label": "c"})), say("done"))
    started = time.perf_counter()
    result = await Agent(client=script, model="m", tools=[slow]).run("go")

    assert time.perf_counter() - started < 0.5
    assert [p.content for p in result.messages[2].parts] == ["a", "b", "c"]
    assert len(result.messages[2].parts) == 3  # all results travel in a single message


async def test_tool_failures_are_reported_to_the_model_not_raised():
    def boom() -> str:
        raise RuntimeError("disk full")

    script = Script(
        calls(("boom", {}), ("add", {"a": "x", "b": 1}), ("nope", {}), ("add", '{"a": 1, "b"')),
        say("recovered"),
    )
    result = await Agent(client=script, model="m", tools=[boom, add]).run("go")

    failure, invalid, unknown, malformed = result.messages[2].parts
    assert all(p.is_error for p in result.messages[2].parts)
    assert failure.content == "RuntimeError: disk full"
    assert "valid integer" in invalid.content
    assert "no tool named 'nope'" in unknown.content and "boom, add" in unknown.content
    assert "not a valid JSON object" in malformed.content
    assert result.output == "recovered"


async def test_max_turns():
    script = Script(*[call("add", {"a": 1, "b": 1}) for _ in range(5)])
    with pytest.raises(MaxTurnsExceeded, match="3 turns"):
        await Agent(client=script, model="m", tools=[add], max_turns=3).run("loop forever")
    assert len(script.requests) == 3


async def test_deps_reach_tools_and_dynamic_instructions():
    @dataclass
    class Deps:
        user_name: str

    def whoami(ctx: Context[Deps]) -> str:
        """Who is asking."""
        return ctx.deps.user_name

    script = Script(call("whoami"), say("You are Ada."))
    agent = Agent(
        client=script, model="m", tools=[whoami], instructions=lambda ctx: f"The user is {ctx.deps.user_name}."
    )

    result = await agent.run("Who am I?", deps=Deps("Ada"))

    assert result.messages[2].parts[0].content == "Ada"
    assert script.requests[0].instructions == "The user is Ada."
    assert script.requests[0].tools[0].parameters["properties"] == {}  # ctx is hidden from the model


async def test_client_and_model_can_be_injected_at_run_time():
    agent = Agent(instructions="No client here.")
    with pytest.raises(FleetError, match="needs a client and a model"):
        await agent.run("Hi")

    script = Script(say("ok"))
    assert (await agent.run("Hi", client=script, model="m")).output == "ok"
    assert script.requests[0].model == "m"


async def test_history_continues_a_conversation_without_mutating_it():
    script = Script(say("Hi Ada."), say("Ada."))
    agent = Agent(client=script, model="m")

    first = await agent.run("I'm Ada.")
    snapshot = list(first.messages)
    second = await agent.run("What's my name?", history=first.messages)

    assert first.messages == snapshot
    assert [m.text for m in script.requests[1].messages] == ["I'm Ada.", "Hi Ada.", "What's my name?"]
    assert len(second.messages) == 4


async def test_messages_survive_a_json_round_trip():
    script = Script(call("add", {"a": 1, "b": 2}), say("3"))
    result = await Agent(client=script, model="m", tools=[add]).run("1+2?")

    stored = json.dumps([m.to_dict() for m in result.messages])
    assert [Message.from_dict(d) for d in json.loads(stored)] == result.messages


# --- structured output ---------------------------------------------------


class City(BaseModel):
    name: str
    population: int


async def test_structured_output_via_final_answer_tool():
    script = Script(call("final_answer", {"name": "Paris", "population": 2_100_000}))
    agent = Agent(client=script, model="m", output_type=City)

    result = await agent.run("Tell me about Paris")

    assert result.output == City(name="Paris", population=2_100_000)
    assert result.finish_reason == "stop"
    offered = {t.name: t for t in script.requests[0].tools}["final_answer"]
    assert offered.parameters["required"] == ["name", "population"]
    assert result.messages[-1].role == "tool"  # the call is answered, so the history can be continued


async def test_invalid_structured_output_is_sent_back_for_another_try():
    script = Script(
        call("final_answer", {"name": "Paris", "population": "lots"}),
        call("final_answer", {"name": "Paris", "population": 2}),
    )
    result = await Agent(client=script, model="m", output_type=City).run("Paris?")

    assert result.output.population == 2
    rejected = script.requests[1].messages[-1].parts[0]
    assert rejected.is_error and "population" in rejected.content


async def test_structured_output_written_as_text_is_accepted_or_nudged():
    script = Script(say('{"name": "Rome", "population": 3}'))
    assert (await Agent(client=script, model="m", output_type=City).run("Rome?")).output.name == "Rome"

    script = Script(say("Rome is lovely."), call("final_answer", {"name": "Rome", "population": 3}))
    assert (await Agent(client=script, model="m", output_type=City).run("Rome?")).output.name == "Rome"
    assert "final_answer" in script.requests[1].messages[-1].text


async def test_non_object_output_types_are_wrapped():
    script = Script(call("final_answer", {"value": ["a", "b"]}))
    agent = Agent(client=script, model="m", output_type=list[str])

    assert (await agent.run("letters")).output == ["a", "b"]
    schema = script.requests[0].tools[0].parameters
    assert schema["properties"]["value"] == {"type": "array", "items": {"type": "string"}}


# --- more than one agent ---------------------------------------------------


async def test_agent_as_tool_runs_in_its_own_conversation():
    sub_script = Script(say("42 widgets"))
    researcher = Agent(name="Researcher", description="Looks things up.", client=sub_script, model="small")
    lead_script = Script(call("researcher", {"task": "Count the widgets"}), say("There are 42."))
    lead = Agent(name="lead", client=lead_script, model="big", tools=[researcher])

    result = await lead.run("How many widgets?")

    assert result.output == "There are 42."
    offered = lead_script.requests[0].tools[0]
    assert (offered.name, offered.description) == ("researcher", "Looks things up.")
    assert [m.text for m in sub_script.requests[0].messages] == ["Count the widgets"]  # nothing of the parent's
    assert result.messages[2].parts[0].content == "42 widgets"
    assert result.usage.requests == 3  # the sub-agent's request is counted


async def test_sub_agent_inherits_client_model_and_deps_from_its_caller():
    def secret(ctx: Context[str]) -> str:
        return ctx.deps

    script = Script(
        call("helper", {"task": "fetch it"}),  # lead
        call("secret"),  # helper
        say("swordfish"),  # helper
        say("It is swordfish."),  # lead
    )
    helper = Agent(name="helper", tools=[secret])
    lead = Agent(name="lead", client=script, model="m", tools=[helper])

    result = await lead.run("What's the secret?", deps="swordfish")

    assert result.output == "It is swordfish."
    assert [r.model for r in script.requests] == ["m"] * 4


async def test_handoff_transfers_the_conversation():
    billing_script = Script(say("Your refund is on its way."))
    billing = Agent(
        name="Billing", description="Handles refunds.", instructions="You do billing.", client=billing_script, model="b"
    )
    triage_script = Script(call("transfer_to_billing"))
    triage = Agent(name="Triage", instructions="Route the user.", client=triage_script, model="t", handoffs=[billing])

    events = []
    result = await triage.run("I want a refund", on_event=events.append)

    assert result.output == "Your refund is on its way."
    assert result.agent is billing
    assert "Handles refunds." in triage_script.requests[0].tools[0].description
    seen = billing_script.requests[0]
    assert seen.instructions == "You do billing."
    assert [m.role for m in seen.messages] == ["user", "assistant", "tool"]  # billing sees the whole conversation
    assert Handoff("Triage", "Billing") in events


async def test_agents_can_hand_off_to_each_other():
    script = Script(call("transfer_to_b"), call("transfer_to_a"), say("back with a"))
    a = Agent(name="a", client=script, model="m")
    b = Agent(name="b", client=script, model="m", handoffs=[a])
    a.handoffs.append(b)

    result = await a.run("ping")
    assert (result.output, result.agent) == ("back with a", a)


def test_duplicate_tool_names_are_rejected_up_front():
    with pytest.raises(ValueError, match="same name"):
        Agent(tools=[add, add])


# --- observing and steering a run ------------------------------------------


async def test_stream_yields_events_then_done():
    script = Script(call("add", {"a": 1, "b": 2}), say("3"))
    agent = Agent(name="calc", client=script, model="m", tools=[add])

    events = [event async for event in agent.stream("1+2?")]

    assert [type(e) for e in events] == [ToolStart, ToolEnd, TextDelta, Done]
    assert events[1].result.content == "3"
    assert events[2] == TextDelta("calc", "3")
    assert events[-1].result.output == "3"


async def test_stream_raises_what_the_run_raises():
    agent = Agent(client=Script(), model="m")
    with pytest.raises(AssertionError, match="ran out of responses"):
        async for _ in agent.stream("hi"):
            pass


async def test_abandoning_a_stream_cancels_the_run():
    finished = []

    async def slow() -> str:
        try:
            await asyncio.sleep(30)
        finally:
            finished.append("cancelled")
        return "never"

    agent = Agent(client=Script(call("slow")), model="m", tools=[slow])
    events = agent.stream("go")
    assert isinstance(await anext(events), ToolStart)
    await events.aclose()
    await asyncio.sleep(0)

    assert finished == ["cancelled"]


async def test_events_from_sub_agents_carry_their_name():
    helper = Agent(name="helper", client=Script(say("sub says hi")), model="m")
    lead = Agent(name="lead", client=Script(call("helper", {"task": "greet"}), say("done")), model="m", tools=[helper])

    deltas = [e async for e in lead.stream("go") if isinstance(e, TextDelta)]
    assert deltas == [TextDelta("helper", "sub says hi"), TextDelta("lead", "done")]


async def test_approve_can_deny_tool_calls():
    ran = []

    def delete(path: str) -> str:
        ran.append(path)
        return "deleted"

    async def approve(ctx, tool_call):
        return tool_call.arguments["path"].startswith("/tmp/") or "Only /tmp may be touched."

    script = Script(calls(("delete", {"path": "/tmp/x"}), ("delete", {"path": "/etc/passwd"})), say("ok"))
    result = await Agent(client=script, model="m", tools=[delete]).run("clean up", approve=approve)

    allowed, denied = result.messages[2].parts
    assert ran == ["/tmp/x"]
    assert (allowed.content, allowed.is_error) == ("deleted", False)
    assert (denied.content, denied.is_error) == ("Only /tmp may be touched.", True)


async def test_truncated_tool_calls_are_not_run():
    ran = []

    def write(text: str = "") -> str:
        ran.append(text)
        return "written"

    script = Script(call("write", {}, finish_reason="length"), say("ok"))
    result = await Agent(client=script, model="m", tools=[write]).run("write a lot")

    assert ran == []
    assert result.messages[2].parts[0].is_error
    assert "output-token limit" in result.messages[2].parts[0].content
    assert result.output == "ok"


async def test_finish_reason_is_reported():
    assert (
        await Agent(client=Script(say("half an ans", finish_reason="length")), model="m").run("?")
    ).finish_reason == "length"
    assert (
        await Agent(client=Script(say("I can't help.", finish_reason="refusal")), model="m").run("?")
    ).finish_reason == "refusal"
    with pytest.raises(FleetError, match="declined"):
        await Agent(client=Script(say("No.", finish_reason="refusal")), model="m", output_type=City).run("?")


def test_run_sync():
    assert Agent(client=Script(say("hi")), model="m").run_sync("hello").output == "hi"


async def test_one_agent_serves_concurrent_runs():
    def echo(request):
        return say(request.messages[-1].text.upper())

    agent = Agent(client=Script(echo, echo, echo), model="m")
    results = await asyncio.gather(*(agent.run(word) for word in ["a", "b", "c"]))
    assert [r.output for r in results] == ["A", "B", "C"]


async def test_replace_swaps_the_client_for_a_test_double():
    agent = Agent(name="prod", client=object(), model="m")
    result = await dataclasses.replace(agent, client=Script(say("fake"))).run("hi")
    assert result.output == "fake"


async def test_input_may_be_messages():
    script = Script(say("ok"))
    await Agent(client=script, model="m").run([user("one"), user("two")])
    assert [m.text for m in script.requests[0].messages] == ["one", "two"]
