"""
Base Provider - Abstract interface for LLM providers.

This allows Fleet to work uniformly across different LLM APIs.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from fleet.payload.payload import Payload
import logging

logger = logging.getLogger(__name__)


class BaseProvider(ABC):
    """
    Abstract base class for LLM providers.

    All providers (OpenAI, Anthropic, OpenRouter, etc.) implement this interface.
    """

    def __init__(self, client: Any, provider_name: str):
        """
        Initialize the provider.

        Args:
            client: The actual client object (openai.Client, anthropic.Anthropic, etc.)
            provider_name: Name of the provider for logging
        """
        self.client = client
        self.provider_name = provider_name
        logger.info(f"Initialized {provider_name} provider")

    @abstractmethod
    def send_message(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        temperature: float = 0.0,
        max_tokens: int = 1024,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: str = "auto",
        **kwargs
    ) -> Payload:
        """
        Send a message and get a response.

        Args:
            model: Model identifier
            messages: List of message dicts
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            tools: Optional list of tool schemas
            tool_choice: How to use tools ("auto", "required", "none")
            **kwargs: Provider-specific options

        Returns:
            Payload with the response
        """
        pass

    @abstractmethod
    def list_available_models(self) -> List[str]:
        """
        List available models for this provider.

        Returns:
            List of model identifiers
        """
        pass

    @abstractmethod
    def validate_model(self, model: str) -> bool:
        """
        Check if a model is available.

        Args:
            model: Model identifier

        Returns:
            True if model is available
        """
        pass

    @abstractmethod
    def format_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Format tool schemas for this provider's API.

        Args:
            tools: Generic tool schemas

        Returns:
            Provider-specific tool schemas
        """
        pass

    def supports_streaming(self) -> bool:
        """Whether this provider supports streaming responses"""
        return False

    def supports_json_mode(self) -> bool:
        """Whether this provider supports JSON mode"""
        return False

    def supports_vision(self) -> bool:
        """Whether this provider supports vision/image inputs"""
        return False

    def __str__(self) -> str:
        return f"{self.provider_name}Provider"

    def __repr__(self) -> str:
        return self.__str__()
