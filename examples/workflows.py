"""Fixed pipelines are plain Python.

When you know the steps in advance, you do not need a model to decide what
happens next. Agents are values and `run` is a coroutine, so the usual tools
apply: a loop for a chain, `asyncio.gather` for a fan-out.

This is what fleet 0.1's `Fleet.compose("synchronously")` and
`Fleet.compose("asynchronously")` did, without a class to learn.

    export OPENAI_API_KEY=...
    python examples/workflows.py
"""

import asyncio

from openai import AsyncOpenAI
from pydantic import BaseModel

from fleet import Agent

client = AsyncOpenAI()
MODEL = "gpt-5"


async def chain(topic: str) -> str:
    """Each step's output is the next step's input."""
    steps = [
        Agent(name="outliner", instructions="Write a five-point outline for an article on the topic given."),
        Agent(name="writer", instructions="Write a 200-word article from the outline given."),
        Agent(name="editor", instructions="Tighten the article given. Return only the final text."),
    ]
    text = topic
    for step in steps:
        text = (await step.run(text, client=client, model=MODEL)).output
    return text


class Verdict(BaseModel):
    approve: bool
    concerns: list[str]


async def fan_out(proposal: str) -> str:
    """Several reviewers look at the same input at once, then one agent combines their findings."""
    reviewers = [
        Agent(name=role, instructions=f"Review the proposal strictly from a {role} point of view.", output_type=Verdict)
        for role in ("security", "cost", "usability")
    ]
    runs = await asyncio.gather(*(reviewer.run(proposal, client=client, model=MODEL) for reviewer in reviewers))

    findings = "\n\n".join(f"{run.agent.name}: {run.output.model_dump_json()}" for run in runs)
    summariser = Agent(instructions="Combine the reviews into one recommendation, in three sentences.")
    return (await summariser.run(f"Proposal:\n{proposal}\n\nReviews:\n{findings}", client=client, model=MODEL)).output


async def main() -> None:
    print(await chain("why small libraries age better than big frameworks"))
    print()
    print(await fan_out("Store user sessions in a public S3 bucket to save on database costs."))


if __name__ == "__main__":
    asyncio.run(main())
