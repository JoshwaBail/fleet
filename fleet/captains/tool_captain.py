"""
ToolCaptain - Agent with function calling capabilities.

Use ToolCaptain when your agent needs to use tools/instruments
to accomplish tasks.
"""

from typing import Optional, Dict, Any, List
from fleet.captains.base_captain import BaseCaptain
from fleet.providers.base_provider import BaseProvider
from fleet.tools.toolbox import Toolbox
from fleet.tools.tool import Tool
from fleet.payload.payload import Payload
import json
import logging

logger = logging.getLogger(__name__)


class ToolCaptain(BaseCaptain):
    """
    ToolCaptain is an agent that can use instruments (tools) to accomplish tasks.

    Perfect for:
    - Agents that need to call APIs
    - Agents that need to access external data
    - Agents that need to perform calculations
    - Any task requiring function calling
    """

    def __init__(
        self,
        provider: BaseProvider,
        name: str = "ToolCaptain",
        system_prompt: str = "You are a helpful AI assistant with access to tools.",
        description: str = "",
        color: str = "white",
        toolbox: Optional[Toolbox] = None,
        tools: Optional[List[Tool]] = None,
        default_model: Optional[str] = None,
        default_temperature: float = 0.0,
        default_max_tokens: int = 2048,
        max_tool_rounds: int = 5
    ):
        """
        Initialize a ToolCaptain.

        Args:
            provider: The LLM provider to use
            name: Name of this captain
            system_prompt: System instructions
            description: Description of this captain's role
            color: Color for terminal output
            toolbox: Toolbox of instruments to use
            tools: Individual instruments (will be added to toolbox)
            default_model: Default model to use
            default_temperature: Default temperature
            default_max_tokens: Default max tokens
            max_tool_rounds: Maximum rounds of tool calling (prevents infinite loops)
        """
        super().__init__(provider, name, system_prompt, description, color)

        # Initialize toolbox
        self.toolbox = toolbox or Toolbox(name=f"{name}_toolbox")

        # Add individual instruments if provided
        if tools:
            for instrument in tools:
                self.toolbox.add_tool(instrument)

        self.default_model = default_model
        self.default_temperature = default_temperature
        self.default_max_tokens = default_max_tokens
        self.max_tool_rounds = max_tool_rounds

        logger.info(f"Initialized ToolCaptain: {name} with {len(self.toolbox)} instruments")

    def add_instrument(self, instrument: Tool):
        """Add an instrument to this captain's toolbox"""
        self.toolbox.add_tool(instrument)
        logger.info(f"{self.name} - Added instrument: {instrument.name}")

    def add_instruments(self, tools: List[Tool]):
        """Add multiple instruments to this captain's toolbox"""
        for instrument in tools:
            self.add_instrument(instrument)

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
        Send a message with tool support.

        This handles the full tool calling loop:
        1. Send message with available tools
        2. If LLM wants to call tools, execute them
        3. Send tool results back
        4. Repeat until LLM provides final response

        Args:
            content: The message content to send
            model: Model identifier
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            tool_choice: How to use tools ("auto", "required", "none")
            **kwargs: Additional provider-specific options

        Returns:
            Payload with the final response
        """
        # Add user message
        self.add_message("user", content)

        # Prepare tools based on provider type
        tools = self._get_provider_tools()

        # Tool calling loop
        rounds = 0
        while rounds < self.max_tool_rounds:
            rounds += 1
            logger.debug(f"{self.name} - Tool round {rounds}/{self.max_tool_rounds}")

            # Send to provider
            payload = self.provider.send_message(
                model=model,
                messages=self.messages,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools if tool_choice != "none" else None,
                tool_choice=tool_choice,
                **kwargs
            )

            # If no tool calls, we're done
            if not payload.tool_calls:
                logger.debug(f"{self.name} - No tool calls, finishing")
                if payload.content:
                    self.add_message("assistant", payload.content)
                payload.captain_name = self.name
                return payload

            # Execute tool calls
            logger.info(f"{self.name} - Executing {len(payload.tool_calls)} tool calls")
            self._handle_tool_calls(payload.tool_calls)

        # Max rounds reached
        logger.warning(f"{self.name} - Max tool rounds ({self.max_tool_rounds}) reached")

        # Make one final call without tools to get a response
        final_payload = self.provider.send_message(
            model=model,
            messages=self.messages,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=None,
            **kwargs
        )

        if final_payload.content:
            self.add_message("assistant", final_payload.content)

        final_payload.captain_name = self.name
        return final_payload

    def _get_provider_tools(self) -> List[Dict[str, Any]]:
        """Get tools formatted for the current provider"""
        provider_name = self.provider.provider_name.lower()

        if "openai" in provider_name or "openrouter" in provider_name:
            return self.toolbox.to_openai_schemas()
        elif "anthropic" in provider_name:
            return self.toolbox.to_anthropic_schemas()
        else:
            # Default to OpenAI format
            return self.toolbox.to_openai_schemas()

    def _handle_tool_calls(self, tool_calls: List[Dict[str, Any]]):
        """
        Execute tool calls and add results to message history.

        Args:
            tool_calls: List of tool calls from the LLM
        """
        # Add assistant message with tool calls
        self.add_message("assistant", content=None, tool_calls=tool_calls)

        # Execute each tool call
        for tool_call in tool_calls:
            function_name = tool_call["function"]["name"]
            function_args_str = tool_call["function"]["arguments"]

            try:
                # Parse arguments
                function_args = json.loads(function_args_str) if isinstance(function_args_str, str) else function_args_str

                # Execute the instrument
                logger.info(f"{self.name} - Executing: {function_name}")
                result = self.toolbox.execute_tool(function_name, **function_args)

                # Convert result to string
                result_str = json.dumps(result) if not isinstance(result, str) else result

            except Exception as e:
                logger.error(f"{self.name} - Error executing {function_name}: {str(e)}")
                result_str = json.dumps({"error": str(e)})

            # Add tool result to messages
            self.add_message(
                "tool",
                content=result_str,
                tool_call_id=tool_call["id"]
            )

    def chat(
        self,
        message: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tool_choice: str = "auto",
        **kwargs
    ) -> Payload:
        """
        Convenience method for sending messages with default parameters.

        Args:
            message: The message to send
            model: Model to use (uses default if not specified)
            temperature: Sampling temperature (uses default if not specified)
            max_tokens: Max tokens (uses default if not specified)
            tool_choice: How to use tools
            **kwargs: Additional options

        Returns:
            Payload with the response
        """
        model = model or self.default_model
        if model is None:
            raise ValueError("No model specified and no default model set")

        temperature = temperature if temperature is not None else self.default_temperature
        max_tokens = max_tokens if max_tokens is not None else self.default_max_tokens

        return self.send_message(
            content=message,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            tool_choice=tool_choice,
            **kwargs
        )

    def __str__(self) -> str:
        return f"ToolCaptain({self.name}, {len(self.toolbox)} tools)"
