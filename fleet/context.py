"""The run context: what a run injects into tools and dynamic instructions."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from .messages import Message, ToolCall, Usage

if TYPE_CHECKING:
    from .agent import Agent
    from .events import Event

D = TypeVar("D")

#: Decides whether a tool call may run. Return ``True`` to allow it; ``False``
#: or a string (the reason, shown to the model) to deny it. May be async.
Approve = Callable[["Context[Any]", ToolCall], "bool | str | Awaitable[bool | str]"]


@dataclass
class Context(Generic[D]):
    """State shared by everything taking part in one run.

    A tool receives it by declaring a parameter annotated ``Context`` (or
    ``Context[MyDeps]``); that parameter is injected and hidden from the model.
    Sub-agents run with a copy that shares ``deps`` and ``usage`` but has its
    own ``agent`` and ``messages``.
    """

    #: Whatever you passed as ``deps=``: a database handle, the current user…
    deps: D = None  # type: ignore[assignment]
    #: Running token total for the whole run, sub-agents included.
    usage: Usage = field(default_factory=Usage)
    #: The agent currently in control.
    agent: Agent[Any] | None = None
    #: The live conversation of the current agent.
    messages: list[Message] = field(default_factory=list)
    #: Run-level defaults for agents that do not set their own.
    client: Any = None
    model: str | None = None
    on_event: Callable[[Event], None] | None = None
    approve: Approve | None = None

    def emit(self, event: Event) -> None:
        if self.on_event is not None:
            self.on_event(event)
