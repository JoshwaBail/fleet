"""
ChatCaptain - Simple conversational agent without tools.

Use ChatCaptain when you need a straightforward conversational agent
that doesn't require function calling or tool usage.
"""

from typing import Optional
from fleet.captains.base_captain import BaseCaptain
from fleet.providers.base_provider import BaseProvider
from fleet.payload.payload import Payload
import logging

logger = logging.getLogger(__name__)


class ChatCaptain(BaseCaptain):
    """
    ChatCaptain is a simple conversational agent.

    Perfect for:
    - Basic Q&A
    - Text generation
    - Analysis and reasoning
    - Any task that doesn't require function calling
    """

    def __init__(
        self,
        provider: BaseProvider,
        name: str = "ChatCaptain",
        system_prompt: str = "You are a helpful AI assistant.",
        description: str = "",
        color: str = "white",
        default_model: Optional[str] = None,
        default_temperature: float = 0.7,
        default_max_tokens: int = 2048
    ):
        """
        Initialize a ChatCaptain.

        Args:
            provider: The LLM provider to use
            name: Name of this captain
            system_prompt: System instructions
            description: Description of this captain's role
            color: Color for terminal output
            default_model: Default model to use
            default_temperature: Default temperature
            default_max_tokens: Default max tokens
        """
        super().__init__(provider, name, system_prompt, description, color)

        self.default_model = default_model
        self.default_temperature = default_temperature
        self.default_max_tokens = default_max_tokens

        logger.info(f"Initialized ChatCaptain: {name}")

    def chat(
        self,
        message: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Payload:
        """
        Send a chat message and get a response.

        This is a convenience method that uses default parameters if not specified.

        Args:
            message: The message to send
            model: Model to use (uses default if not specified)
            temperature: Sampling temperature (uses default if not specified)
            max_tokens: Max tokens (uses default if not specified)
            **kwargs: Additional provider-specific options

        Returns:
            Payload with the response
        """
        # Use defaults if not specified
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
            **kwargs
        )

    def get_response(self, message: str, model: str, **kwargs) -> str:
        """
        Convenience method to get just the response content as a string.

        Args:
            message: The message to send
            model: Model to use
            **kwargs: Additional options

        Returns:
            Response content as string
        """
        payload = self.chat(message, model, **kwargs)
        return payload.content

    def __str__(self) -> str:
        return f"ChatCaptain({self.name})"
