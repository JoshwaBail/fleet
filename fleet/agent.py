"""The agent and its loop.

An :class:`Agent` is a description: instructions, tools, and who it can hand
off to. It holds no conversation state, so one agent can serve many runs at
once. Running it is a loop: ask the model, run the tools it asks for, repeat
until it answers.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import re
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Generic, TypeVar, cast

from pydantic import TypeAdapter, ValidationError

from .context import Approve, Context
from .events import Done, Event, Handoff, TextDelta, ToolEnd, ToolStart
from .messages import Message, ToolCall, ToolResult, Usage, user
from .providers import Request, resolve
from .tools import Tool, render, strip_titles, tool

T = TypeVar("T")
logger = logging.getLogger("fleet")

#: The tool an agent with an ``output_type`` calls to deliver its result.
FINAL_ANSWER = "final_answer"


class FleetError(Exception):
    """Base class for errors raised by fleet itself."""


class MaxTurnsExceeded(FleetError):
    """The agent was still calling tools when it ran out of turns."""


@dataclass(slots=True)
class Result(Generic[T]):
    #: The answer: text, or an instance of the agent's ``output_type``.
    output: T
    #: The whole conversation. Pass it as ``history=`` to continue it.
    messages: list[Message]
    usage: Usage
    #: The agent that produced the answer, which differs from the one you
    #: started if there was a handoff. Continue the conversation with this one.
    agent: Agent[Any]
    #: ``"stop"`` normally; ``"length"`` if the answer was cut off by the
    #: output-token limit; ``"refusal"`` if the model declined.
    finish_reason: str = "stop"


@dataclass(kw_only=True, eq=False)
class Agent(Generic[T]):
    name: str = "agent"
    #: The system prompt, or a function of the run context returning one.
    instructions: str | Callable[[Context[Any]], str | Awaitable[str]] = ""
    #: The model client *you* construct: an OpenAI-compatible or Anthropic
    #: client, sync or async, or any :class:`fleet.providers.Provider`. May be
    #: left unset and supplied per run instead.
    client: Any = None
    model: str | None = None
    #: Functions, :class:`Tool` objects, or other agents (delegated to as sub-agents).
    tools: Sequence[Any] = ()
    #: Agents this one may pass the whole conversation to.
    handoffs: list[Agent[Any]] = field(default_factory=list)
    #: Makes the run return a validated instance of this type instead of text.
    output_type: type[T] | None = None
    #: What this agent is for, as seen by an agent that delegates or hands off to it.
    description: str = ""
    #: Extra keyword arguments for the provider's API call, passed through
    #: untouched: ``{"temperature": 0}``, ``{"thinking": {"type": "adaptive"}}``…
    params: dict[str, Any] = field(default_factory=dict)
    #: Model requests allowed in one run before :class:`MaxTurnsExceeded`.
    max_turns: int = 20

    def __post_init__(self) -> None:
        self.tools = [_as_tool(item) for item in self.tools]
        self._output = _Output(self.output_type) if self.output_type not in (None, str) else None
        self._toolbox()  # fail now, not mid-run, on a name clash

    async def run(
        self,
        input: str | Message | Sequence[Message],
        *,
        history: Sequence[Message] = (),
        deps: Any = None,
        client: Any = None,
        model: str | None = None,
        on_event: Callable[[Event], None] | None = None,
        approve: Approve | None = None,
    ) -> Result[T]:
        """Run the agent until it produces an answer.

        Args:
            input: The new user message (or messages).
            history: An earlier conversation to continue, usually ``result.messages``.
            deps: Anything your tools need; it reaches them as ``Context.deps``.
            client: Default client for agents in this run that have none of their own.
            model: Default model name, likewise.
            on_event: Called with each :mod:`fleet.events` event as it happens.
            approve: Consulted before every tool call; see :data:`fleet.Approve`.
        """
        new = [user(input)] if isinstance(input, str) else [input] if isinstance(input, Message) else list(input)
        ctx = Context(
            deps=deps, messages=[*history, *new], client=client, model=model, on_event=on_event, approve=approve
        )
        return await _loop(self, ctx)

    def run_sync(self, input: str | Message | Sequence[Message], **kwargs: Any) -> Result[T]:
        """:meth:`run`, for code that is not async. Takes the same arguments."""
        return asyncio.run(self.run(input, **kwargs))

    async def stream(self, input: str | Message | Sequence[Message], **kwargs: Any) -> AsyncIterator[Event]:
        """:meth:`run`, yielding events as they happen. The last one is :class:`Done`."""
        queue: asyncio.Queue[Event | None] = asyncio.Queue()
        task = asyncio.create_task(self.run(input, on_event=queue.put_nowait, **kwargs))
        task.add_done_callback(lambda _: queue.put_nowait(None))
        try:
            while (event := await queue.get()) is not None:
                yield event
            result = task.result()  # re-raises if the run failed
            yield Done(result.agent.name, result)
        finally:
            task.cancel()  # stop the run if the consumer stopped listening

    def as_tool(self, *, name: str | None = None, description: str | None = None) -> Tool:
        """This agent as a tool: the caller writes it a task, it works in a fresh conversation and reports back."""

        async def delegate(ctx: Context[Any], arguments: dict[str, Any]) -> Any:
            if not arguments.get("task"):
                raise ValueError("`task` is required")
            result = await _loop(self, replace(ctx, messages=[user(str(arguments["task"]))]))
            return result.output

        parameters = {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "What to do, with all the context needed: this agent cannot see your conversation.",
                }
            },
            "required": ["task"],
            "additionalProperties": False,
        }
        return Tool(
            name or _slug(self.name),
            description or self.description or f"Delegate a task to {self.name}.",
            parameters,
            delegate,
        )

    def _toolbox(self) -> tuple[dict[str, Tool], dict[str, Agent[Any]]]:
        """Everything the model can call this turn, and which of those names are handoffs."""
        tools = [*cast("list[Tool]", self.tools)]
        handoffs = {f"transfer_to_{_slug(target.name)}": target for target in self.handoffs}
        for tool_name, target in handoffs.items():
            description = f"Hand the conversation over to {target.name}. {target.description}".strip()
            tools.append(Tool(tool_name, description, {"type": "object", "properties": {}}, _unreachable))
        if self._output:
            tools.append(self._output.tool)
        by_name = {t.name: t for t in tools}
        if len(by_name) != len(tools):
            names = sorted(t.name for t in tools)
            raise ValueError(f"Agent {self.name!r} has more than one tool with the same name: {names}")
        return by_name, handoffs


async def _loop(agent: Agent[Any], ctx: Context[Any]) -> Result[Any]:
    """Run ``agent`` on ``ctx.messages`` until it answers. Shared by top-level runs and sub-agents."""
    turns = 0
    while True:
        if turns >= agent.max_turns:
            raise MaxTurnsExceeded(f"{agent.name!r} did not finish within {agent.max_turns} turns")
        turns += 1
        ctx.agent = agent
        # An agent without its own client or model uses the run's, or else those of
        # the agent that delegated or handed off to it.
        client = ctx.client = agent.client or ctx.client
        model = ctx.model = agent.model or ctx.model
        if client is None or model is None:
            raise FleetError(
                f"Agent {agent.name!r} needs a client and a model: set them on the agent or pass them to run()"
            )

        tools, handoffs = agent._toolbox()
        instructions = agent.instructions
        if callable(instructions):
            instructions = await _resolve(instructions(ctx))
        request = Request(model, instructions, ctx.messages, list(tools.values()), agent.params)
        on_text = (lambda text, name=agent.name: ctx.emit(TextDelta(name, text))) if ctx.on_event else None
        response = await resolve(client).complete(request, on_text)
        ctx.usage.add(response.usage)
        ctx.messages.append(response.message)
        calls, reason = response.message.tool_calls, response.finish_reason

        if reason in ("length", "refusal") and calls:
            # The reply was cut short, so its tool calls may be incomplete: answer them without running anything.
            why = "the output-token limit was reached" if reason == "length" else "the model declined to continue"
            results = [ToolResult(c.id, f"Not run: {why} before this call was complete.", is_error=True) for c in calls]
            ctx.messages.append(Message("tool", [*results]))
            if reason == "length":
                continue
        if reason == "refusal":
            if agent._output:
                raise FleetError(f"{agent.name!r} declined to answer: {response.message.text or '(no explanation)'}")
            return _result(response.message.text, agent, ctx, reason)

        if not calls:
            if agent._output is None:
                return _result(response.message.text, agent, ctx, reason)
            try:  # A model may also write the JSON out directly instead of calling the tool.
                return _result(agent._output.adapter.validate_json(response.message.text), agent, ctx, reason)
            except ValidationError:
                ctx.messages.append(user(f"Give your final answer by calling the `{FINAL_ANSWER}` tool."))
                continue

        outcomes = await asyncio.gather(*(_call(c, tools, handoffs, ctx) for c in calls))
        ctx.messages.append(Message("tool", [result for result, _ in outcomes]))
        for call, (result, value) in zip(calls, outcomes, strict=True):
            if call.name == FINAL_ANSWER and not result.is_error:
                return _result(value, agent, ctx, reason)
        for call, (result, _) in zip(calls, outcomes, strict=True):
            if call.name in handoffs and not result.is_error:
                ctx.emit(Handoff(agent.name, handoffs[call.name].name))
                agent = handoffs[call.name]
                break


def _result(output: Any, agent: Agent[Any], ctx: Context[Any], reason: str) -> Result[Any]:
    # A run that ends by calling `final_answer` stopped normally, whatever the provider called it.
    return Result(output, ctx.messages, ctx.usage, agent, "stop" if reason == "tool_calls" else reason)


async def _call(
    call: ToolCall, tools: dict[str, Tool], handoffs: dict[str, Agent[Any]], ctx: Context[Any]
) -> tuple[ToolResult, Any]:
    """Run one tool call. Failures become error results, so the model can see them and recover."""
    name = cast("Agent[Any]", ctx.agent).name
    ctx.emit(ToolStart(name, call))
    value: Any = None
    try:
        if isinstance(call.arguments, str):
            raise ValueError(f"arguments were not a valid JSON object: {call.arguments!r}")
        if call.name not in tools:
            raise LookupError(f"no tool named {call.name!r}; available tools: {', '.join(tools)}")
        verdict: bool | str = True
        if ctx.approve is not None and call.name != FINAL_ANSWER:
            verdict = await _resolve(ctx.approve(ctx, call))
        if verdict is not True:
            reason = verdict if isinstance(verdict, str) and verdict else "This tool call was not approved."
            result = ToolResult(call.id, reason, is_error=True)
        elif call.name in handoffs:
            result = ToolResult(call.id, f"Transferred to {handoffs[call.name].name}.")
        else:
            value = await tools[call.name].handler(ctx, call.arguments)
            result = ToolResult(call.id, "Answer accepted." if call.name == FINAL_ANSWER else render(value))
    except Exception as error:
        logger.debug("tool %r failed", call.name, exc_info=True)
        result = ToolResult(call.id, f"{type(error).__name__}: {error}", is_error=True)
    ctx.emit(ToolEnd(name, call, result))
    return result, value


class _Output:
    """Structured output, delivered through a tool call so it works the same on every provider."""

    def __init__(self, output_type: Any) -> None:
        self.adapter: TypeAdapter[Any] = TypeAdapter(output_type)
        schema = strip_titles(self.adapter.json_schema())
        # Tool parameters must be a JSON object; anything else (a list, an int) travels inside one.
        wrapped = schema.get("type") != "object" or "properties" not in schema
        if wrapped:
            definitions = schema.pop("$defs", None)
            schema = {"type": "object", "properties": {"value": schema}, "required": ["value"]}
            if definitions:
                schema["$defs"] = definitions

        async def validate(ctx: Context[Any], arguments: dict[str, Any]) -> Any:
            return self.adapter.validate_python(arguments.get("value") if wrapped else arguments)

        description = (
            "Deliver your final answer. Call this once the task is complete, instead of answering in plain text."
        )
        self.tool = Tool(FINAL_ANSWER, description, schema, validate)


def _as_tool(item: Any) -> Tool:
    if isinstance(item, Tool):
        return item
    if isinstance(item, Agent):
        return item.as_tool()
    if callable(item):
        return tool(item)
    raise TypeError(f"Not a tool: {item!r}. Expected a function, a Tool, or an Agent.")


async def _unreachable(ctx: Context[Any], arguments: dict[str, Any]) -> Any:
    raise AssertionError("handoffs are handled by the loop")


async def _resolve(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


def _slug(name: str) -> str:
    """A model-safe tool name derived from an agent's name."""
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", name).strip("_").lower() or "agent"
