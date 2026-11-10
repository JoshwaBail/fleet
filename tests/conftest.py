"""A scripted provider: the agent loop, with no model and no network."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable
from typing import Any

import pytest

from fleet import Message, Request, Response, Text, ToolCall, Usage


def say(text: str, **kwargs: Any) -> Response:
    """A model reply that is just text."""
    return Response(Message("assistant", [Text(text)]), Usage(1, 10, 5), **kwargs)


def call(name: str, arguments: Any = None, *, id: str | None = None, **kwargs: Any) -> Response:
    """A model reply that calls one tool."""
    return calls((name, arguments if arguments is not None else {}, id), **kwargs)


def calls(*specs: Any, **kwargs: Any) -> Response:
    """A model reply that calls several tools at once. Each spec is ``(name, arguments[, id])``."""
    parts = [
        ToolCall(spec[2] if len(spec) > 2 and spec[2] else f"call_{i}", spec[0], spec[1])
        for i, spec in enumerate(specs)
    ]
    kwargs.setdefault("finish_reason", "tool_calls")
    return Response(Message("assistant", list(parts)), Usage(1, 10, 5), **kwargs)


class Script:
    """Replies with a fixed sequence of responses and records every request it receives.

    A step may also be a function of the request, for replies that depend on it.
    """

    def __init__(self, *steps: Response | Callable[[Request], Response]) -> None:
        self.steps = list(steps)
        self.requests: list[Request] = []

    async def complete(self, request: Request, on_text: Callable[[str], None] | None = None) -> Response:
        # Snapshot the conversation: the loop keeps appending to the list it passed.
        self.requests.append(
            Request(request.model, request.instructions, list(request.messages), request.tools, request.params)
        )
        if not self.steps:
            raise AssertionError("the script ran out of responses")
        step = self.steps.pop(0)
        response = step(request) if callable(step) else step
        if on_text and response.message.text:
            on_text(response.message.text)
        return response


def pytest_pyfunc_call(pyfuncitem: pytest.Function) -> bool | None:
    """Run ``async def`` tests on a fresh event loop, without needing a plugin."""
    if not inspect.iscoroutinefunction(pyfuncitem.obj):
        return None
    arguments = {name: pyfuncitem.funcargs[name] for name in pyfuncitem._fixtureinfo.argnames}
    asyncio.run(pyfuncitem.obj(**arguments))
    return True
