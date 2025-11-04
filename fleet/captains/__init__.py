"""Captains module - Agent implementations for Fleet"""

from fleet.captains.base_captain import BaseCaptain
from fleet.captains.chat_captain import ChatCaptain
from fleet.captains.tool_captain import ToolCaptain

# Advanced Patterns
from fleet.captains.navigator import Navigator
from fleet.captains.quartermaster import Quartermaster
from fleet.captains.admiral import Admiral
from fleet.captains.watch_change import WatchChange

__all__ = [
    "BaseCaptain",
    "ChatCaptain",
    "ToolCaptain",
    # Advanced Patterns
    "Navigator",  # ReAct pattern
    "Quartermaster",  # Reflection pattern
    "Admiral",  # Plan-and-Execute pattern
    "WatchChange",  # Handoff pattern
]
