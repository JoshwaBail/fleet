"""
BaseCaptain - The foundation for all Fleet agents.

A Captain is an autonomous agent that can navigate conversations,
make decisions, and optionally use instruments (tools) to accomplish tasks.
"""

from typing import List, Dict, Any, Optional, Union
from fleet.payload.payload import Payload
from fleet.providers.base_provider import BaseProvider
from fleet.instruments.arsenal import Arsenal
import logging

logger = logging.getLogger(__name__)


class BaseCaptain:
    """
    BaseCaptain is the foundation for all agent types in Fleet.

    A Captain manages:
    - Conversation history (transmissions)
    - System prompt (mission orders)
    - Provider communication
    - Optional tool/instrument usage
    """

    def __init__(
        self,
        provider: BaseProvider,
        name: str = "Captain",
        system_prompt: str = "You are a helpful AI assistant.",
        description: str = "",
        color: str = "white"
    ):
        """
        Initialize a Captain.

        Args:
            provider: The LLM provider to use
            name: Name of this captain
            system_prompt: System instructions (mission orders)
            description: Description of this captain's role
            color: Color for terminal output (for Fleet compositions)
        """
        self.provider = provider
        self.name = name
        self.system_prompt = system_prompt
        self.description = description
        self.color = color

        # Message history (transmissions)
        self.messages: List[Dict[str, Any]] = []

        # Add system message
        self._initialize_messages()

        logger.info(f"Initialized Captain: {name}")

    def _initialize_messages(self):
        """Initialize message history with system prompt"""
        if self.system_prompt:
            self.messages.append({
                "role": "system",
                "content": self.system_prompt
            })

    def add_message(self, role: str, content: str, **kwargs):
        """
        Add a message to the conversation history.

        Args:
            role: Message role ("user", "assistant", "system", "tool")
            content: Message content
            **kwargs: Additional message properties
        """
        message = {
            "role": role,
            "content": content,
            **kwargs
        }
        self.messages.append(message)
        logger.debug(f"{self.name} - Added {role} message")

    def get_messages(self) -> List[Dict[str, Any]]:
        """Get conversation history"""
        return self.messages

    def clear_messages(self):
        """Clear conversation history and reinitialize with system prompt"""
        logger.info(f"{self.name} - Clearing conversation history")
        self.messages = []
        self._initialize_messages()

    def get_last_message(self) -> Optional[Dict[str, Any]]:
        """Get the most recent message"""
        return self.messages[-1] if self.messages else None

    def send_message(
        self,
        content: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        **kwargs
    ) -> Payload:
        """
        Send a message and get a response.

        This is the basic version without tool support.
        Subclasses can override for more complex behavior.

        Args:
            content: The message content to send
            model: Model identifier
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional provider-specific options

        Returns:
            Payload with the response
        """
        # Add user message
        self.add_message("user", content)

        # Send to provider
        logger.info(f"{self.name} - Sending message to {model}")
        payload = self.provider.send_message(
            model=model,
            messages=self.messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )

        # Add assistant response to history
        if payload.content:
            self.add_message("assistant", payload.content)

        # Set captain name in payload
        payload.captain_name = self.name

        return payload

    def __str__(self) -> str:
        return f"Captain({self.name})"

    def __repr__(self) -> str:
        return self.__str__()
