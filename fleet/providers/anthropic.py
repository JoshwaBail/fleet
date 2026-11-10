"""Adapter for the Anthropic Messages API."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from typing import Any

from ..messages import Message, Part, Text, ToolCall, Usage
from . import Request, Response, is_async

#: The Messages API requires ``max_tokens``. This is roomy enough not to cut
#: answers short and within every current model's limit; raise it per agent
#: with ``params={"max_tokens": ...}``.
MAX_TOKENS = 16_000

_FINISH = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "tool_use": "tool_calls",
    "max_tokens": "length",
    "refusal": "refusal",
}


class Messages:
    """Wraps any Anthropic client (``Anthropic``, ``AsyncAnthropic``, and the cloud-platform variants).

    Requests are always streamed, even when nobody is listening for text: the
    SDK refuses long non-streaming requests, and an agent's turns can be long.
    """

    def __init__(self, client: Any) -> None:
        self._client = client
        self._async = is_async(client.messages.create)

    async def complete(self, request: Request, on_text: Callable[[str], None] | None = None) -> Response:
        kwargs: dict[str, Any] = {
            "model": request.model,
            "max_tokens": MAX_TOKENS,
            "messages": _encode(request.messages),
        }
        if request.instructions:
            kwargs["system"] = request.instructions
        if request.tools:
            kwargs["tools"] = [
                {"name": t.name, "description": t.description, "input_schema": t.parameters} for t in request.tools
            ]
        kwargs.update(request.params)
        # Beta features are opted into with `betas`, which only the beta endpoint accepts.
        api = self._client.beta.messages if "betas" in kwargs else self._client.messages

        if self._async:
            async with api.stream(**kwargs) as stream:
                if on_text is not None:
                    async for piece in stream.text_stream:
                        on_text(piece)
                return _decode(await stream.get_final_message())

        loop = asyncio.get_running_loop()

        def pump() -> Any:
            with api.stream(**kwargs) as stream:
                if on_text is not None:
                    for piece in stream.text_stream:
                        loop.call_soon_threadsafe(on_text, piece)
                return stream.get_final_message()

        return _decode(await asyncio.to_thread(pump))


def _encode(messages: Sequence[Message]) -> list[dict[str, Any]]:
    wire: list[dict[str, Any]] = []
    for message in messages:
        if message.role == "assistant" and message.raw and message.raw.get("provider") == "anthropic":
            # Our own earlier turn: send it back exactly as received, so that
            # thinking blocks and the like reach the API unmodified.
            content = message.raw["content"]
        else:
            content = [_block(part) for part in message.parts if not (isinstance(part, Text) and not part.text)]
        if content:
            wire.append({"role": "assistant" if message.role == "assistant" else "user", "content": content})
    return wire


def _block(part: Part) -> dict[str, Any]:
    if isinstance(part, Text):
        return {"type": "text", "text": part.text}
    if isinstance(part, ToolCall):
        arguments = part.arguments if isinstance(part.arguments, dict) else {}
        return {"type": "tool_use", "id": part.id, "name": part.name, "input": arguments}
    block: dict[str, Any] = {"type": "tool_result", "tool_use_id": part.id, "content": part.content}
    if part.is_error:
        block["is_error"] = True
    return block


def _decode(message: Any) -> Response:
    parts: list[Part] = []
    for block in message.content:
        if block.type == "text":
            parts.append(Text(block.text))
        elif block.type == "tool_use":
            parts.append(ToolCall(block.id, block.name, block.input))
    # Serialised the way the SDK itself serialises a response block used as a request param.
    content = [
        block.model_dump(
            mode="json", by_alias=True, exclude_unset=True, exclude=getattr(block, "__api_exclude__", None)
        )
        for block in message.content
    ]
    usage = message.usage
    cached = getattr(usage, "cache_read_input_tokens", 0) or 0
    written = getattr(usage, "cache_creation_input_tokens", 0) or 0
    return Response(
        Message("assistant", parts, raw={"provider": "anthropic", "content": content}),
        Usage(
            requests=1,
            # Anthropic reports uncached, cache-read and cache-write tokens separately.
            input_tokens=(usage.input_tokens or 0) + cached + written,
            output_tokens=usage.output_tokens or 0,
            cached_tokens=cached,
        ),
        _FINISH.get(message.stop_reason or "end_turn", message.stop_reason or "stop"),
    )
