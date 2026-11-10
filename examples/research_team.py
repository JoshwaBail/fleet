"""A lead agent that delegates to sub-agents, streamed as it happens.

The lead never sees the sub-agents' working, only their answers, which keeps its
context small. The sub-agents name no client, so they use the lead's, and run on
a cheaper model.

    export ANTHROPIC_API_KEY=...
    python examples/research_team.py "Compare SQLite and DuckDB for analytics"
"""

import asyncio
import sys

from anthropic import AsyncAnthropic

from fleet import Agent, Done, TextDelta, ToolEnd, ToolStart

NOTES = {
    "sqlite": "Row-oriented, transactional, single writer. Excellent for point lookups and small writes.",
    "duckdb": "Column-oriented, vectorised execution. Built for aggregations over large tables.",
}


def lookup(topic: str) -> str:
    """Look a topic up in the reference notes.

    Args:
        topic: A single keyword, such as a product name.
    """
    return NOTES.get(topic.lower(), f"Nothing on {topic!r}. Known topics: {', '.join(NOTES)}.")


researcher = Agent(
    name="researcher",
    description="Looks up facts. Give it one focused question and it returns what it found.",
    instructions="Answer from the reference notes only. Say so plainly if they do not cover the question.",
    model="claude-haiku-5-5",
    tools=[lookup],
)

critic = Agent(
    name="critic",
    description="Reviews a draft for unsupported claims and gaps. Give it the full draft.",
    instructions="List concrete problems with the draft, most serious first. Reply 'No issues.' if there are none.",
    model="claude-haiku-5-5",
)

lead = Agent(
    name="lead",
    client=AsyncAnthropic(),
    model="claude-opus-5-5",
    instructions=(
        "You write short, accurate briefings. Ask the researcher for the facts you need, "
        "asking about separate topics in parallel. Have the critic review your draft before you answer."
    ),
    tools=[researcher, critic],
    params={"betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"},
)


async def main(question: str) -> None:
    async for event in lead.stream(question):
        match event:
            case TextDelta(agent="lead", text=text):
                print(text, end="", flush=True)
            case ToolStart(agent=agent, call=call):
                print(f"\n  [{agent}] {call.name}({call.arguments})")
            case ToolEnd(agent=agent, call=call, result=result) if result.is_error:
                print(f"\n  [{agent}] {call.name} failed: {result.content}")
            case Done(result=result):
                usage = result.usage
                print(f"\n\n{usage.requests} requests, {usage.input_tokens} tokens in, {usage.output_tokens} out")


if __name__ == "__main__":
    asyncio.run(main(" ".join(sys.argv[1:]) or "Compare SQLite and DuckDB for analytics."))
