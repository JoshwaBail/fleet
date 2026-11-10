"""An agent for JobFair, built on its MCP server.

JobFair writes job adverts and audits them for inclusive language. Its MCP
server supplies the tools. This file supplies the agent around them:

- every JobFair tool, discovered from the server at start-up;
- one local tool, to save exported documents to disk;
- a budget for the two tools that cost a JobFair credit, enforced in code, with
  a person asked first when there is one.

It runs two ways: as a chat, or unattended over a set of advert files, where it
returns one typed result per file.

    pip install -e ".[anthropic]" "mcp>=2"
    export ANTHROPIC_API_KEY=...
    export JOBFAIR_API_KEY=jf_...     # JobFair web app -> API keys

    python examples/jobfair.py                  # chat
    python examples/jobfair.py adverts/*.md     # audit each file
"""

import asyncio
import os
import sys
import urllib.request
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from pathlib import Path

import httpx2
from anthropic import AsyncAnthropic
from mcp import Client
from mcp.client.streamable_http import streamable_http_client
from pydantic import BaseModel

from fleet import Agent, Context, Event, Message, TextDelta, ToolCall, ToolStart
from fleet.mcp import mcp_tools

URL = os.environ.get("JOBFAIR_MCP_URL", "https://api-railway.job-fair.uk/mcp")

#: JobFair charges one credit for each completed run of these. Everything else is free.
BILLED = {"analyze_job_description", "generate_job_description"}


@dataclass
class Session:
    """Injected into a run as `deps`: what it may spend, and whether a person is there to ask."""

    credits: int
    interactive: bool


async def approve(ctx: Context[Session], call: ToolCall) -> bool | str:
    """Let free tools through; hold billed ones to the session's budget."""
    if call.name not in BILLED:
        return True
    if ctx.deps.credits <= 0:
        return "This session's JobFair credit budget is spent. Tell the user; do not try again."
    ctx.deps.credits -= 1  # reserve before waiting on anyone, so parallel calls cannot overspend
    if ctx.deps.interactive:
        answer = await asyncio.to_thread(input, f"\n  {call.name} costs one JobFair credit. Go ahead? [y/N] ")
        if answer.strip().lower() != "y":
            ctx.deps.credits += 1
            return "The user declined to spend a credit on this."
    return True


def save_export(download_url: str, filename: str) -> str:
    """Save an exported document into the local `exports` folder.

    Args:
        download_url: The signed `download_url` returned by export_job_description.
        filename: The `filename` returned alongside it.
    """
    if not download_url.startswith("https://"):
        raise ValueError("Only https download URLs are accepted.")
    target = Path("exports") / Path(filename).name  # the name only: never write outside exports/
    target.parent.mkdir(exist_ok=True)
    urllib.request.urlretrieve(download_url, target)
    return f"Saved {target} ({target.stat().st_size} bytes)."


@asynccontextmanager
async def jobfair_agent() -> AsyncIterator[Agent[str]]:
    """Connect to JobFair and build an agent whose tools are whatever the server offers."""
    # An analysis can take a couple of minutes, hence the long read timeout.
    http = httpx2.AsyncClient(
        headers={"X-API-Key": os.environ["JOBFAIR_API_KEY"]}, timeout=httpx2.Timeout(30, read=300)
    )
    async with http, Client(streamable_http_client(URL, http_client=http)) as mcp:
        yield Agent(
            name="jobfair",
            client=AsyncAnthropic(),
            model="claude-opus-5-5",
            # The server describes how it wants to be used; that is most of the prompt.
            instructions=f"{mcp.instructions or ''}\n\nAfter an export, save the file with save_export and say where it went.",
            tools=[*await mcp_tools(mcp), save_export],
            # Passed straight to the Messages API: if the model declines a request, retry it on another model.
            params={"betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"},
        )


def show(event: Event) -> None:
    if isinstance(event, TextDelta):
        print(event.text, end="", flush=True)
    elif isinstance(event, ToolStart):
        print(f"\n  [{event.call.name}]")


async def chat(agent: Agent[str]) -> None:
    session, history = Session(credits=5, interactive=True), list[Message]()
    while (text := input("\nyou> ").strip()) not in ("", "quit", "exit"):
        result = await agent.run(text, history=history, deps=session, approve=approve, on_event=show)
        history = result.messages
        print()


class Audit(BaseModel):
    job_title: str
    quality_score: str  # as JobFair reports it
    top_findings: list[str]  # the two or three that matter most, in plain language
    request_id: str  # exports of this audit are free for 30 days


UNATTENDED = """

You are running unattended, so there is nobody to answer questions. Analyse the advert you are given once, passing
save_to_library=false so that nothing is written to a workspace library. Do not export anything."""


async def audit(agent: Agent[str], paths: list[str]) -> None:
    """Audit each file in parallel, spending at most one credit per file."""
    auditor = replace(agent, output_type=Audit, instructions=agent.instructions + UNATTENDED)
    session = Session(credits=len(paths), interactive=False)
    prompts = [f"Audit this advert.\n\n<advert>\n{Path(path).read_text()}\n</advert>" for path in paths]
    runs = await asyncio.gather(
        *(auditor.run(prompt, deps=session, approve=approve) for prompt in prompts), return_exceptions=True
    )
    for path, run in zip(paths, runs, strict=True):
        if isinstance(run, BaseException):
            print(f"{path}: failed: {run}")
            continue
        result: Audit = run.output
        print(f"{path}: {result.job_title}, quality {result.quality_score} (request {result.request_id})")
        for finding in result.top_findings:
            print(f"  - {finding}")


async def main() -> None:
    async with jobfair_agent() as agent:
        await (audit(agent, sys.argv[1:]) if len(sys.argv) > 1 else chat(agent))


if __name__ == "__main__":
    asyncio.run(main())
