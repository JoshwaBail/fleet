"""Captains module - Agent implementations for Fleet"""

from fleet.captains.base_captain import BaseCaptain
from fleet.captains.chat_captain import ChatCaptain
from fleet.captains.tool_captain import ToolCaptain

# Advanced Patterns
from fleet.captains.react_captain import ReActCaptain
from fleet.captains.reflective_captain import ReflectiveCaptain
from fleet.captains.admiral import Admiral
from fleet.captains.handoff_captain import HandoffCaptain

__all__ = [
    "BaseCaptain",
    "ChatCaptain",
    "ToolCaptain",
    # Advanced Patterns
    "ReActCaptain",  # ReAct pattern (formerly Navigator)
    "ReflectiveCaptain",  # Reflection pattern (formerly Quartermaster)
    "Admiral",  # Plan-and-Execute pattern
    "HandoffCaptain",  # Handoff pattern (formerly WatchChange)
]
