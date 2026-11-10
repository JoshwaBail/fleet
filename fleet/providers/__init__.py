"""The seam between the agent loop and a model API.

A provider is anything with an async ``complete`` method. fleet ships two thin
adapters around clients *you* construct, and picks between them by looking at
the client you pass:

- ``chat.completions`` (OpenAI, Azure OpenAI, Groq, Ollama, OpenRouter, and
  anything else speaking the Chat Completions dialect)
  -> :class:`fleet.providers.openai.ChatCompletions`
- ``messages`` (Anthropic, including its Bedrock, Vertex and Foundry clients)
  -> :class:`fleet.providers.anthropic.Messages`

To support another API, or to script a model in tests, implement
:class:`Provider` yourself and pass that object as ``client=``.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from ..messages import Message, Usage
from ..tools import Tool


@dataclass(frozen=True, slots=True)
class Request:
    model: str
    instructions: str
    messages: Sequence[Message]
    tools: Sequence[Tool]
    #: Extra keyword arguments for the provider's API call, passed through as-is.
    params: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Response:
    message: Message
    usage: Usage = field(default_factory=lambda: Usage(requests=1))
    #: ``"stop"``, ``"tool_calls"``, ``"length"`` (hit the output-token limit),
    #: ``"refusal"``, or a provider-specific value passed through.
    finish_reason: str = "stop"


@runtime_checkable
class Provider(Protocol):
    async def complete(self, request: Request, on_text: Callable[[str], None] | None = None) -> Response:
        """Make one model request.

        If ``on_text`` is given, call it with pieces of the reply text as they
        arrive (or once with the whole text if the API cannot stream).
        """
        ...


def resolve(client: Any) -> Provider:
    """Return ``client`` as a :class:`Provider`, wrapping a known SDK client if needed."""
    if isinstance(client, Provider):
        return client
    if hasattr(getattr(client, "chat", None), "completions"):
        from .openai import ChatCompletions

        return ChatCompletions(client)
    if hasattr(getattr(client, "messages", None), "stream"):
        from .anthropic import Messages

        return Messages(client)
    raise TypeError(
        f"Don't know how to talk to {type(client).__name__!r}. Pass an OpenAI-compatible client, "
        "an Anthropic client, or an object with an async `complete(request, on_text)` method."
    )


def is_async(method: Callable[..., Any]) -> bool:
    """Whether an SDK method belongs to an async client (the SDKs wrap their coroutines)."""
    return inspect.iscoroutinefunction(inspect.unwrap(method))
