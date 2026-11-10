# fleet

A small agent framework. Bring your own client.

```python
from anthropic import Anthropic
from fleet import Agent


def get_weather(city: str) -> str:
    """Look up the current weather in a city."""
    return f"18°C and sunny in {city}"


agent = Agent(
    client=Anthropic(),  # your client, configured your way
    model="claude-opus-5-5",
    instructions="You are a concise assistant.",
    tools=[get_weather],  # plain functions
)

print(agent.run_sync("Should I take an umbrella in Lisbon?").output)
```

That is the whole idea:

- **You construct the client.** fleet never reads an API key, picks a base URL, or
  retries behind your back. Pass an OpenAI-compatible client or an Anthropic one,
  sync or async, and fleet works out how to talk to it.
- **Tools are functions.** The signature, type hints and docstring are the schema.
- **An agent is a description, a run is a loop.** Ask the model, run the tools it
  asks for, repeat until it answers. The loop is one function of about sixty lines
  in [`fleet/agent.py`](fleet/agent.py), and you can read it.

The only dependency is `pydantic`.

## Install

Not on PyPI yet, so install from a checkout:

```bash
pip install -e ".[anthropic]"
```

Swap `anthropic` for `openai`, or name both. Python 3.10 or newer.

## The pieces

### Agents

```python
agent = Agent(
    name="support",
    instructions="You help customers with their orders.",
    client=client,
    model="gpt-5",
    tools=[find_order, refund],
    params={"reasoning_effort": "low"},
)
```

`params` is handed to the provider's API call exactly as written. fleet does not
wrap sampling settings, token limits, reasoning controls or anything else, so
whatever your provider supports today works, and so will whatever it adds next.

An agent holds no conversation state. The same object can serve any number of
concurrent runs.

There are three ways to run one, taking the same arguments:

```python
result = await agent.run("Where is order 1042?")  # async
result = agent.run_sync("Where is order 1042?")  # in a script
async for event in agent.stream("Where is order 1042?"):  # as it happens
    ...
```

A `Result` has the `output`, the full `messages`, the token `usage`, the `agent`
that answered, and a `finish_reason`.

To continue a conversation, pass the earlier messages back:

```python
first = await agent.run("My order number is 1042.")
second = await agent.run("Where is it?", history=first.messages)
```

Messages are plain dataclasses. `message.to_dict()` and `Message.from_dict()`
turn them into JSON and back, so storing a conversation is up to you and takes
two lines.

### Tools

Any function is a tool. Type hints become the JSON Schema, the docstring becomes
the description, and a Google-style `Args:` section describes each parameter.
Arguments are validated with pydantic before your function is called, so a tool
that asks for an `int` or a `datetime` or a pydantic model gets one.

```python
from typing import Literal


async def search_orders(customer_email: str, status: Literal["open", "shipped", "any"] = "any") -> list[Order]:
    """Find a customer's orders.

    Args:
        customer_email: The email address on the account.
        status: Only return orders in this state.
    """
    ...
```

Things worth knowing:

- Tools may be sync or async. Sync tools run in a worker thread, so a slow one
  does not stall the others.
- When the model asks for several tools in one turn, they run concurrently.
- If a tool raises, or is called with bad arguments, the error text is sent back
  to the model so that it can correct itself. The run does not crash. You see
  these as `ToolEnd` events with `result.is_error` set.
- Return anything. Strings go to the model as they are, everything else as JSON.

Use `@tool(name=..., description=...)` to override what is inferred, or build a
`Tool` by hand to expose something that is not a Python function.

### Dependencies

The same principle that applies to the client applies to your tools: they should
be handed what they need, not reach for globals. Pass anything as `deps=` and a
tool receives it by declaring a `Context` parameter, which is hidden from the
model.

```python
from dataclasses import dataclass
from fleet import Agent, Context


@dataclass
class Deps:
    db: Database
    user_id: str


def my_orders(ctx: Context[Deps]) -> list[Order]:
    """List the current user's orders."""
    return ctx.deps.db.orders_for(ctx.deps.user_id)


agent = Agent(
    tools=[my_orders],
    instructions=lambda ctx: f"You are helping user {ctx.deps.user_id}.",
)

await agent.run("What did I order?", deps=Deps(db, "u_17"), client=client, model="gpt-5")
```

Note that this agent names no client. `client` and `model` can be left off the
agent and supplied per run, which lets you define agents at import time and
inject the client where the request is handled. An agent's own client, when it
has one, takes precedence.

Bound methods are tools too, which is often all the injection you need:
`Agent(tools=[inventory.count, inventory.reserve])`.

### Structured output

```python
from pydantic import BaseModel


class Invoice(BaseModel):
    vendor: str
    total: float
    line_items: list[str]


extractor = Agent(client=client, model="gpt-5", output_type=Invoice)
invoice = (await extractor.run(email_text)).output  # an Invoice
```

`output_type` can be a pydantic model, a dataclass, or any type pydantic can
validate, such as `list[str]`. The agent is given a `final_answer` tool whose
schema is that type, so this works the same way on every provider and alongside
your other tools. If the model's answer does not validate, it is told why and
tries again.

### A fleet of agents

There are two ways for agents to work together, and they are different things.

**Delegation.** Put an agent in another agent's `tools`. The lead agent writes
it a task, the sub-agent works in a fresh conversation of its own, and only its
answer comes back. This keeps the lead's context small and lets sub-agents use a
cheaper model. Several delegations in one turn run in parallel.

```python
researcher = Agent(
    name="researcher",
    description="Finds facts on the web. Give it one focused question.",
    model="claude-haiku-5-5",
    tools=[web_search],
)
writer = Agent(name="writer", description="Turns notes into clean prose.", model="claude-haiku-5-5")

lead = Agent(
    name="lead",
    client=client,
    model="claude-opus-5-5",
    instructions="Break the request down, delegate, then assemble the final answer.",
    tools=[researcher, writer],
)
```

Sub-agents that name no client or model inherit them from the agent that called
them, so the client is set once at the top.

**Handoff.** List agents in `handoffs` and the agent can pass the whole
conversation to one of them, which then answers the user directly.

```python
billing = Agent(name="billing", description="Refunds and invoices.", tools=[refund])
tech = Agent(name="tech", description="Bugs and outages.", tools=[check_status])
triage = Agent(name="triage", instructions="Route the user to the right team.", handoffs=[billing, tech])

result = await triage.run("I was charged twice", client=client, model="gpt-5")
result.agent  # billing: continue the conversation with this one
```

Because messages are provider-neutral, a conversation can be handed between
agents on different providers.

When you want a fixed pipeline, not a model deciding what happens next, write it
as Python. See [`examples/workflows.py`](examples/workflows.py).

### Watching a run

`stream` yields events as the run proceeds, sub-agents included. Each event
names the agent it came from.

```python
from fleet import TextDelta, ToolStart, ToolEnd, Handoff, Done

async for event in lead.stream("Write a briefing on solid-state batteries"):
    match event:
        case TextDelta(agent="lead", text=text):
            print(text, end="", flush=True)
        case ToolStart(agent=agent, call=call):
            print(f"\n[{agent}] {call.name}({call.arguments})")
        case Done(result=result):
            print(f"\n{result.usage.input_tokens} in, {result.usage.output_tokens} out")
```

If you only want a callback, `run(..., on_event=fn)` delivers the same events.

### Approving tool calls

`approve` is consulted before every tool call in the run. Return `True` to let
it proceed, or `False` or a reason to refuse. A refusal is reported to the model,
which can then try something else.

```python
def approve(ctx, call):
    if call.name == "refund" and call.arguments["amount"] > 100:
        return "Refunds over £100 need a human. Tell the customer you have escalated it."
    return True


await agent.run("Refund my £250 order", approve=approve)
```

It may be async, so it can wait on a person.

### MCP

Connect to an MCP server however you like and pass the client in:

```python
from mcp import Client
from fleet.mcp import mcp_tools

async with Client("http://localhost:8000/mcp") as mcp:
    agent = Agent(client=client, model="gpt-5", tools=await mcp_tools(mcp))
    await agent.run("What's in the inventory?")
```

## Providers

fleet picks an adapter by the shape of the client you pass:

| Your client has | Adapter | Covers |
| --- | --- | --- |
| `chat.completions` | [`ChatCompletions`](fleet/providers/openai.py) | OpenAI, Azure OpenAI, and anything speaking the same dialect: Groq, Ollama, OpenRouter, vLLM |
| `messages` | [`Messages`](fleet/providers/anthropic.py) | Anthropic, and its Bedrock, Vertex and Foundry clients |

For anything else, a provider is one async method. Implement it and pass the
object as `client=`:

```python
from fleet import Message, Request, Response, Text


class Echo:
    async def complete(self, request: Request, on_text=None) -> Response:
        return Response(Message("assistant", [Text(request.messages[-1].text)]))
```

That is also how you test an agent without a network. The test suite's
[`Script`](tests/conftest.py) is about twenty lines.

### Notes for Claude

Everything in `params` reaches `messages.stream()` unchanged, so Claude's
features are available without fleet knowing about them:

```python
Agent(
    client=AsyncAnthropic(),
    model="claude-opus-5-5",
    params={
        "max_tokens": 64_000,  # fleet's default is 16,000
        "output_config": {"effort": "high"},  # this model's own default is medium
        "thinking": {"type": "adaptive", "display": "summarized"},
        "cache_control": {"type": "ephemeral"},  # prompt caching; a large saving in a tool loop
        "betas": ["server-side-fallback-2026-07-01"],
        "fallbacks": "default",  # re-run on another model if this one declines
    },
)
```

If `params` contains `betas`, the request goes to the beta endpoint. Thinking
blocks are returned to the API untouched on each turn of a tool loop, as the API
requires. Requests are always streamed, because the SDK rejects long
non-streaming ones.

## What is not here

Deliberately small means some things are left out for now:

- Images and files in messages. Messages carry text and tool calls.
- Trimming or summarising a conversation that outgrows the context window.
- Server-side tools, such as a provider's hosted web search.
- OpenAI's Responses API. The Chat Completions adapter covers OpenAI.
- Retries and rate limiting, which belong to the client you construct.

## Coming from fleet 0.1

Version 0.2 is a rewrite and shares no API with 0.1.

| 0.1 | 0.2 |
| --- | --- |
| `Agent(client, system_prompt, name, ...)` | `Agent(client=..., model=..., instructions=..., name=...)` |
| `agent.send_message(model, message, function_schemas, ...)` | `agent.run(text)`; the model lives on the agent |
| `functions=` plus hand-written `function_schemas=` | `tools=[fn]`; the schema is inferred |
| `json_mode=`, `json_schema=` | `output_type=` |
| `temperature=`, `max_tokens=` arguments | `params={...}`, passed through |
| Messages stored on the agent | `result.messages`, passed back as `history=` |
| `Fleet(...).compose("asynchronously")` with a synthesis step | A lead agent with the others in `tools` |
| `Fleet(...).compose("synchronously")` | A `for` loop; see `examples/workflows.py` |
| `ContextAgent` | Removed |

The Groq client still works; it is handled as an OpenAI-compatible client.

## Development

```bash
uv sync
uv run pytest
uv run ruff check . && uv run mypy
```

The provider tests drive the real OpenAI and Anthropic SDKs against a fake HTTP
transport, so they check the exact requests fleet produces without needing a key
or a network.
