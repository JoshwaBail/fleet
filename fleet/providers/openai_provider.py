"""
OpenAI Provider - Implementation for OpenAI API.
"""

from typing import List, Dict, Any, Optional
from fleet.providers.base_provider import BaseProvider
from fleet.payload.payload import Payload
import openai
import json
import logging

logger = logging.getLogger(__name__)


class OpenAIProvider(BaseProvider):
    """
    Provider implementation for OpenAI API.

    Supports true dependency injection - pass either an api_key or a pre-configured client.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        client: Optional[openai.Client] = None,
        base_url: Optional[str] = None,
        organization: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize OpenAI provider.

        Args:
            api_key: OpenAI API key (if not providing a client)
            client: Pre-configured OpenAI client (takes precedence over api_key)
            base_url: Optional base URL for API
            organization: Optional organization ID
            **kwargs: Additional arguments passed to openai.Client
        """
        if client is None:
            if api_key is None:
                raise ValueError("Must provide either 'api_key' or 'client'")

            client_kwargs = {"api_key": api_key, **kwargs}
            if base_url:
                client_kwargs["base_url"] = base_url
            if organization:
                client_kwargs["organization"] = organization

            client = openai.Client(**client_kwargs)

        super().__init__(client, "OpenAI")
        self._cached_models = None

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
        """Send a message using OpenAI API"""

        # Prepare request parameters
        request_params = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs
        }

        # Add tools if provided
        if tools:
            request_params["tools"] = tools
            request_params["tool_choice"] = tool_choice

        logger.debug(f"OpenAI request: model={model}, messages={len(messages)}, tools={len(tools) if tools else 0}")

        try:
            response = self.client.chat.completions.create(**request_params)

            # Extract content
            message = response.choices[0].message
            content = message.content
            tool_calls = message.tool_calls

            # Create tool calls list if present
            tool_calls_list = None
            if tool_calls:
                tool_calls_list = [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in tool_calls
                ]

            payload = Payload(
                content=content,
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
                tool_calls=tool_calls_list,
                metadata={
                    "model": model,
                    "finish_reason": response.choices[0].finish_reason,
                    "provider": "openai"
                }
            )

            logger.debug(f"OpenAI response: {payload.total_tokens} tokens")
            return payload

        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            raise

    def list_available_models(self) -> List[str]:
        """List available OpenAI models"""
        if self._cached_models is None:
            try:
                models = self.client.models.list()
                self._cached_models = [model.id for model in models.data]
            except Exception as e:
                logger.error(f"Failed to list models: {str(e)}")
                # Return common models as fallback
                self._cached_models = [
                    "gpt-4o",
                    "gpt-4o-mini",
                    "gpt-4-turbo",
                    "gpt-4",
                    "gpt-3.5-turbo"
                ]
        return self._cached_models

    def validate_model(self, model: str) -> bool:
        """Check if a model is available"""
        available = self.list_available_models()
        return model in available

    def format_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Format tools for OpenAI API.

        OpenAI expects tools in this format:
        {
            "type": "function",
            "function": {
                "name": "...",
                "description": "...",
                "parameters": {...}
            }
        }
        """
        # Tools should already be in the correct format from Tool.to_openai_schema()
        return tools

    def supports_streaming(self) -> bool:
        return True

    def supports_json_mode(self) -> bool:
        return True

    def supports_vision(self) -> bool:
        return True
