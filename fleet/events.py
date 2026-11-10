"""What a run reports while it is happening.

Pass ``on_event=`` to ``Agent.run`` or iterate ``Agent.stream``. Every event
names the agent it came from, so sub-agent activity can be told apart.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .messages import ToolCall, ToolResult

if TYPE_CHECKING:
    from .agent import Result


@dataclass(frozen=True, slots=True)
class TextDelta:
    """A piece of the model's reply."""

    agent: str
    text: str


@dataclass(frozen=True, slots=True)
class ToolStart:
    agent: str
    call: ToolCall


@dataclass(frozen=True, slots=True)
class ToolEnd:
    agent: str
    call: ToolCall
    result: ToolResult


@dataclass(frozen=True, slots=True)
class Handoff:
    """Control of the conversation passed from ``agent`` to ``target``."""

    agent: str
    target: str


@dataclass(frozen=True, slots=True)
class Done:
    """The last event of ``Agent.stream``: the run's result."""

    agent: str
    result: Result[Any]


Event = TextDelta | ToolStart | ToolEnd | Handoff | Done
