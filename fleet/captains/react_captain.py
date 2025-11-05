"""
ReActCaptain - ReAct (Reasoning and Acting) Pattern

A ReActCaptain uses the ReAct pattern: iteratively reasoning about what to do,
taking actions, and observing the results until the task is complete.

Think → Act → Observe → Repeat
"""

from typing import Optional, List, Dict, Any
from fleet.captains.tool_captain import ToolCaptain
from fleet.providers.base_provider import BaseProvider
from fleet.tools.toolbox import Toolbox
from fleet.tools.tool import Tool
from fleet.payload.payload import Payload
import logging

logger = logging.getLogger(__name__)


class ReActCaptain(ToolCaptain):
    """
    ReActCaptain implements the ReAct (Reasoning and Acting) pattern.

    The ReActCaptain alternates between reasoning (thinking about what to do)
    and acting (using tools or responding), with explicit observation steps.

    Perfect for:
    - Exploratory tasks
    - Trial-and-error problem solving
    - When you need visibility into reasoning
    - Interactive problem-solving
    """

    def __init__(
        self,
        provider: BaseProvider,
        name: str = "ReActCaptain",
        system_prompt: str = None,
        description: str = "",
        color: str = "cyan",
        toolbox: Optional[Toolbox] = None,
        tools: Optional[List[Tool]] = None,
        default_model: Optional[str] = None,
        default_temperature: float = 0.0,
        default_max_tokens: int = 2048,
        max_iterations: int = 10,
        verbose: bool = True
    ):
        """
        Initialize a ReActCaptain (ReAct agent).

        Args:
            provider: The LLM provider
            name: ReActCaptain name
            system_prompt: System instructions (auto-generated if None)
            description: ReActCaptain's role description
            color: Terminal color
            toolbox: Tools available to this navigator
            tools: Individual instruments to add
            default_model: Default model
            default_temperature: Default temperature
            default_max_tokens: Default max tokens
            max_iterations: Maximum reasoning-acting iterations
            verbose: Whether to print reasoning steps
        """
        if system_prompt is None:
            system_prompt = self._create_react_prompt()

        super().__init__(
            provider=provider,
            name=name,
            system_prompt=system_prompt,
            description=description,
            color=color,
            toolbox=toolbox,
            tools=tools,
            default_model=default_model,
            default_temperature=default_temperature,
            default_max_tokens=default_max_tokens,
            max_tool_rounds=max_iterations
        )

        self.max_iterations = max_iterations
        self.verbose = verbose
        self.reasoning_trace = []

        logger.info(f"Initialized ReActCaptain: {name} with ReAct pattern (max_iterations={max_iterations})")

    def _create_react_prompt(self) -> str:
        """Create a ReAct-style system prompt"""
        return """You are a ReActCaptain using the ReAct (Reasoning and Acting) pattern.

For each task, you should:
1. THINK: Reason about what to do next
2. ACT: Take an action (use a tool or provide an answer)
3. OBSERVE: Observe the result
4. Repeat until you can answer the question

When thinking, explain your reasoning step-by-step.
When acting, use tools when needed or provide the final answer.
Be methodical and show your work."""

    def navigate(
        self,
        task: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Payload:
        """
        Navigate through a task using ReAct pattern.

        This wraps the chat method but provides ReAct-specific logging.

        Args:
            task: The task to accomplish
            model: Model to use
            temperature: Sampling temperature
            max_tokens: Max tokens
            **kwargs: Additional options

        Returns:
            Payload with final answer and reasoning trace
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"🧭 ReActCaptain: {self.name}")
            print(f"Task: {task}")
            print(f"{'='*60}\n")

        # Clear previous reasoning trace
        self.reasoning_trace = []

        # Execute using underlying tool captain logic
        result = self.chat(
            message=task,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )

        # Store reasoning trace in metadata
        result.metadata["reasoning_trace"] = self.reasoning_trace
        result.metadata["pattern"] = "ReAct"
        result.metadata["iterations"] = len(self.reasoning_trace)

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"✓ Navigation Complete!")
            print(f"Iterations: {len(self.reasoning_trace)}")
            print(f"{'='*60}\n")

        return result

    def send_message(
        self,
        content: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        tool_choice: str = "auto",
        **kwargs
    ) -> Payload:
        """
        Override to add reasoning trace logging.
        """
        # Log the reasoning step
        iteration = len(self.reasoning_trace) + 1
        self.reasoning_trace.append({
            "iteration": iteration,
            "thought": content,
            "timestamp": __import__('datetime').datetime.now().isoformat()
        })

        if self.verbose:
            print(f"💭 Thought {iteration}: {content[:100]}...")

        # Call parent implementation
        result = super().send_message(
            content=content,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            tool_choice=tool_choice,
            **kwargs
        )

        # Log any tool calls
        if result.tool_calls and self.verbose:
            for tc in result.tool_calls:
                print(f"🔧 Action: {tc['function']['name']}")

        # Log the observation (result)
        if result.content and self.verbose:
            print(f"👁️  Observation: {result.content[:100]}...\n")

        return result

    def __str__(self) -> str:
        return f"ReActCaptain({self.name}, ReAct pattern, {len(self.toolbox)} tools)"
