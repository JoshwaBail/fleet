"""
Handoff Captain - Handoff Pattern

A Handoff Captain captain can transfer control and context to another captain,
like changing watch duty on a ship.

Agent A → Handoff → Agent B (with full context)
"""

from typing import Optional, Dict, List, Any, Callable
from fleet.captains.tool_captain import ToolCaptain
from fleet.providers.base_provider import BaseProvider
from fleet.tools.toolbox import Toolbox
from fleet.tools.tool import Tool, instrument as tool_decorator
from fleet.payload.payload import Payload
import json
import logging

logger = logging.getLogger(__name__)


class HandoffCaptain(ToolCaptain):
    """
    HandoffCaptain implements the Handoff pattern (inspired by OpenAI Swarm).

    A HandoffCaptain captain can hand off conversations to other captains,
    transferring full context seamlessly.

    Perfect for:
    - Conversational AI / chatbots
    - Multi-stage processes (triage → specialist → escalation)
    - When agents need to tag-team
    - Complex workflows with decision points
    """

    def __init__(
        self,
        provider: BaseProvider,
        name: str = "Handoff Captain",
        system_prompt: str = "You are a helpful assistant who can hand off to specialists when needed.",
        description: str = "",
        color: str = "green",
        toolbox: Optional[Toolbox] = None,
        tools: Optional[List[Tool]] = None,
        default_model: Optional[str] = None,
        default_temperature: float = 0.7,
        default_max_tokens: int = 2048,
        handoff_enabled: bool = True
    ):
        """
        Initialize a HandoffCaptain captain (Handoff-capable agent).

        Args:
            provider: The LLM provider
            name: Captain name
            system_prompt: System instructions
            description: Captain's role
            color: Terminal color
            toolbox: Tools available
            tools: Individual instruments
            default_model: Default model
            default_temperature: Default temperature
            default_max_tokens: Default max tokens
            handoff_enabled: Whether handoffs are enabled
        """
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
            default_max_tokens=default_max_tokens
        )

        self.handoff_enabled = handoff_enabled
        self.handoff_targets: Dict[str, 'HandoffCaptain'] = {}
        self.handoff_history: List[Dict] = []

        logger.info(f"Initialized HandoffCaptain: {name} with Handoff pattern")

    def register_handoff(self, target_name: str, target_captain: 'HandoffCaptain'):
        """
        Register another captain as a handoff target.

        Args:
            target_name: Name to identify this handoff target
            target_captain: The captain to hand off to
        """
        self.handoff_targets[target_name] = target_captain

        # Add handoff as an instrument
        if self.handoff_enabled:
            handoff_instrument = self._create_handoff_instrument(target_name, target_captain)
            self.toolbox.add_tool(handoff_instrument)

        logger.info(f"{self.name} registered handoff to {target_captain.name} as '{target_name}'")

    def _create_handoff_instrument(self, target_name: str, target_captain: 'HandoffCaptain') -> Tool:
        """Create an instrument for handing off to another captain"""

        def handoff_function(reason: str = "Transferring to specialist") -> dict:
            """Hand off to another captain"""
            return {
                "handoff": True,
                "target": target_name,
                "target_captain": target_captain.name,
                "reason": reason
            }

        from fleet.tools.tool import Tool, ToolParameter

        return Tool(
            name=f"handoff_to_{target_name}",
            description=f"Hand off this conversation to {target_captain.name} ({target_captain.description})",
            function=handoff_function,
            parameters=[
                ToolParameter(
                    name="reason",
                    type="string",
                    description="Reason for the handoff",
                    required=False,
                    default="Transferring to specialist"
                )
            ]
        )

    def chat(
        self,
        message: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_handoffs: int = 3,
        **kwargs
    ) -> Payload:
        """
        Chat with handoff support.

        Args:
            message: The message
            model: Model to use
            temperature: Sampling temperature
            max_tokens: Max tokens
            max_handoffs: Maximum number of handoffs to prevent loops
            **kwargs: Additional options

        Returns:
            Payload with response and handoff chain
        """
        handoff_chain = [self.name]
        current_captain = self
        current_message = message
        handoff_count = 0

        while handoff_count < max_handoffs:
            # Get response from current captain
            result = super(HandoffCaptain, current_captain).send_message(
                content=current_message,
                model=model or current_captain.default_model,
                temperature=temperature if temperature is not None else current_captain.default_temperature,
                max_tokens=max_tokens if max_tokens is not None else current_captain.default_max_tokens,
                **kwargs
            )

            # Check if a handoff was requested
            if result.tool_calls:
                handoff_requested = False
                for tool_call in result.tool_calls:
                    func_name = tool_call['function']['name']
                    if func_name.startswith('handoff_to_'):
                        # Handoff detected
                        target_name = func_name.replace('handoff_to_', '')

                        if target_name in current_captain.handoff_targets:
                            next_captain = current_captain.handoff_targets[target_name]

                            # Parse args
                            args = json.loads(tool_call['function']['arguments'])
                            reason = args.get('reason', 'Handoff requested')

                            # Log handoff
                            handoff_info = {
                                "from": current_captain.name,
                                "to": next_captain.name,
                                "reason": reason,
                                "message": current_message
                            }
                            current_captain.handoff_history.append(handoff_info)

                            print(f"\n🔄 Handoff: {current_captain.name} → {next_captain.name}")
                            print(f"   Reason: {reason}\n")

                            # Update for next iteration
                            handoff_chain.append(next_captain.name)
                            current_captain = next_captain
                            current_message = f"[Handed off from {handoff_info['from']}]\n\n{current_message}"
                            handoff_count += 1
                            handoff_requested = True
                            break

                if not handoff_requested:
                    # No handoff, we're done
                    break
            else:
                # No tool calls, we're done
                break

        # Add handoff metadata
        result.metadata["handoff_chain"] = handoff_chain
        result.metadata["handoffs_count"] = handoff_count
        result.metadata["pattern"] = "Handoff"

        return result

    def __str__(self) -> str:
        handoff_info = f", {len(self.handoff_targets)} handoff targets" if self.handoff_targets else ""
        return f"HandoffCaptain({self.name}{handoff_info})"
