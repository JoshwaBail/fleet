"""The conversation model: provider-neutral messages, and token usage.

A conversation is a plain ``list[Message]``. Nothing here knows about any
provider; the adapters in :mod:`fleet.providers` translate to and from each
wire format.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal


@dataclass(frozen=True, slots=True)
class Text:
    text: str


@dataclass(frozen=True, slots=True)
class ToolCall:
    id: str
    name: str
    #: Parsed arguments. A ``str`` only when the model emitted malformed JSON,
    #: in which case the call is answered with an error instead of being run.
    arguments: dict[str, Any] | str


@dataclass(frozen=True, slots=True)
class ToolResult:
    id: str  #: The ``ToolCall.id`` this answers.
    content: str
    is_error: bool = False


Part = Text | ToolCall | ToolResult
_PARTS = {cls.__name__: cls for cls in (Text, ToolCall, ToolResult)}


@dataclass(slots=True)
class Message:
    role: Literal["user", "assistant", "tool"]
    parts: list[Part]
    #: Opaque provider payload for an assistant turn. It is replayed verbatim
    #: to the provider that produced it, which is how reasoning blocks and
    #: other provider-specific content survive a tool loop unmodified.
    raw: dict[str, Any] | None = None

    @property
    def text(self) -> str:
        return "".join(p.text for p in self.parts if isinstance(p, Text))

    @property
    def tool_calls(self) -> list[ToolCall]:
        return [p for p in self.parts if isinstance(p, ToolCall)]

    def to_dict(self) -> dict[str, Any]:
        """A JSON-serialisable form, for persisting a conversation."""
        parts = [{"type": type(p).__name__, **asdict(p)} for p in self.parts]
        return {"role": self.role, "parts": parts, "raw": self.raw}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Message:
        parts = [_PARTS[p["type"]](**{k: v for k, v in p.items() if k != "type"}) for p in data["parts"]]
        return cls(data["role"], parts, data.get("raw"))


def user(text: str) -> Message:
    """A user message holding ``text``."""
    return Message("user", [Text(text)])


@dataclass(slots=True)
class Usage:
    """Token accounting, summed over every model request in a run."""

    requests: int = 0
    #: All prompt tokens, including those served from a cache.
    input_tokens: int = 0
    output_tokens: int = 0
    #: The portion of ``input_tokens`` read from the provider's prompt cache.
    cached_tokens: int = 0

    def add(self, other: Usage) -> None:
        self.requests += other.requests
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cached_tokens += other.cached_tokens
