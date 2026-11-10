"""Adapter for the Chat Completions API, as spoken by OpenAI and compatible services."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from typing import Any

from ..messages import Message, Part, Text, ToolCall, ToolResult, Usage
from . import Request, Response, is_async

_FINISH = {
    "stop": "stop",
    "tool_calls": "tool_calls",
    "function_call": "tool_calls",
    "length": "length",
    "content_filter": "refusal",
}


class ChatCompletions:
    """Wraps any client exposing ``chat.completions.create``, sync or async."""

    def __init__(self, client: Any) -> None:
        self._create = client.chat.completions.create
        self._async = is_async(self._create)

    async def complete(self, request: Request, on_text: Callable[[str], None] | None = None) -> Response:
        kwargs: dict[str, Any] = {"model": request.model, "messages": _encode(request)}
        if request.tools:
            kwargs["tools"] = [
                {
                    "type": "function",
                    "function": {"name": t.name, "description": t.description, "parameters": t.parameters},
                }
                for t in request.tools
            ]
        kwargs.update(request.params)

        if on_text is None:
            create = self._create(**kwargs) if self._async else asyncio.to_thread(self._create, **kwargs)
            completion = await create
            choice = completion.choices[0]
            calls = [
                (c.id, c.function.name, c.function.arguments)
                for c in choice.message.tool_calls or []
                if c.type == "function"
            ]
            return _response(choice.message.content or "", calls, choice.finish_reason, completion.usage)

        # Streaming: accumulate the deltas into the same shape as above.
        kwargs["stream"] = True
        kwargs.setdefault("stream_options", {"include_usage": True})
        text: list[str] = []
        calls_by_index: dict[int, list[str]] = {}  # index -> [id, name, arguments]
        state: dict[str, Any] = {"finish": None, "usage": None}

        def feed(chunk: Any, emit: Callable[[str], Any]) -> None:
            state["usage"] = getattr(chunk, "usage", None) or state["usage"]
            for choice in chunk.choices:
                delta = choice.delta
                if delta.content:
                    text.append(delta.content)
                    emit(delta.content)
                for call in delta.tool_calls or []:
                    slot = calls_by_index.setdefault(call.index, ["", "", ""])
                    slot[0] = call.id or slot[0]
                    if call.function:
                        slot[1] += call.function.name or ""
                        slot[2] += call.function.arguments or ""
                state["finish"] = choice.finish_reason or state["finish"]

        if self._async:
            async for chunk in await self._create(**kwargs):
                feed(chunk, on_text)
        else:
            loop = asyncio.get_running_loop()

            def pump() -> None:
                for chunk in self._create(**kwargs):
                    feed(chunk, lambda piece: loop.call_soon_threadsafe(on_text, piece))

            await asyncio.to_thread(pump)

        streamed = [(id, name, arguments) for _, (id, name, arguments) in sorted(calls_by_index.items())]
        return _response("".join(text), streamed, state["finish"], state["usage"])


def _encode(request: Request) -> list[dict[str, Any]]:
    wire: list[dict[str, Any]] = []
    if request.instructions:
        wire.append({"role": "system", "content": request.instructions})
    for message in request.messages:
        if message.role == "tool":
            wire += [
                {"role": "tool", "tool_call_id": p.id, "content": p.content}
                for p in message.parts
                if isinstance(p, ToolResult)
            ]
        elif message.role == "assistant":
            entry: dict[str, Any] = {"role": "assistant", "content": message.text}
            if message.tool_calls:
                entry["content"] = message.text or None
                entry["tool_calls"] = [
                    {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": _dumps(c.arguments)}}
                    for c in message.tool_calls
                ]
            wire.append(entry)
        else:
            wire.append({"role": "user", "content": message.text})
    return wire


def _dumps(arguments: dict[str, Any] | str) -> str:
    return arguments if isinstance(arguments, str) else json.dumps(arguments)


def _loads(arguments: str) -> dict[str, Any] | str:
    """Parse tool arguments, keeping the raw string if the model wrote invalid JSON."""
    try:
        parsed = json.loads(arguments or "{}")
    except json.JSONDecodeError:
        return arguments
    return parsed if isinstance(parsed, dict) else arguments


def _response(text: str, calls: list[tuple[str, str, str]], finish: str | None, usage: Any) -> Response:
    parts: list[Part] = [Text(text)] if text else []
    parts += [ToolCall(id, name, _loads(arguments)) for id, name, arguments in calls]
    details = getattr(usage, "prompt_tokens_details", None)
    return Response(
        Message("assistant", parts),
        Usage(
            requests=1,
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
            cached_tokens=getattr(details, "cached_tokens", 0) or 0,
        ),
        _FINISH.get(finish or "stop", finish or "stop"),
    )
