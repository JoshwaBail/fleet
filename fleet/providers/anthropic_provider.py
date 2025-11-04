"""
Anthropic Provider - Implementation for Anthropic/Claude API.
"""

from typing import List, Dict, Any, Optional
from fleet.providers.base_provider import BaseProvider
from fleet.payload.payload import Payload
import anthropic
import json
import logging

logger = logging.getLogger(__name__)


class AnthropicProvider(BaseProvider):
    """
    Provider implementation for Anthropic/Claude API.

    Supports true dependency injection - pass either an api_key or a pre-configured client.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        client: Optional[anthropic.Anthropic] = None,
        base_url: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize Anthropic provider.

        Args:
            api_key: Anthropic API key (if not providing a client)
            client: Pre-configured Anthropic client (takes precedence over api_key)
            base_url: Optional base URL for API
            **kwargs: Additional arguments passed to anthropic.Anthropic
        """
        if client is None:
            if api_key is None:
                raise ValueError("Must provide either 'api_key' or 'client'")

            client_kwargs = {"api_key": api_key, **kwargs}
            if base_url:
                client_kwargs["base_url"] = base_url

            client = anthropic.Anthropic(**client_kwargs)

        super().__init__(client, "Anthropic")

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
        """Send a message using Anthropic API"""

        # Anthropic separates system messages from the message list
        system_message = None
        user_messages = []

        for msg in messages:
            if msg.get("role") == "system":
                system_message = msg.get("content", "")
            else:
                user_messages.append(msg)

        # Prepare request parameters
        request_params = {
            "model": model,
            "messages": user_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs
        }

        if system_message:
            request_params["system"] = system_message

        # Add tools if provided
        if tools:
            request_params["tools"] = tools
            if tool_choice == "required":
                request_params["tool_choice"] = {"type": "any"}
            elif tool_choice != "auto":
                request_params["tool_choice"] = {"type": tool_choice}

        logger.debug(f"Anthropic request: model={model}, messages={len(user_messages)}, tools={len(tools) if tools else 0}")

        try:
            response = self.client.messages.create(**request_params)

            # Extract content
            content_blocks = response.content
            text_content = ""
            tool_calls_list = []

            for block in content_blocks:
                if block.type == "text":
                    text_content += block.text
                elif block.type == "tool_use":
                    tool_calls_list.append({
                        "id": block.id,
                        "type": "function",
                        "function": {
                            "name": block.name,
                            "arguments": json.dumps(block.input)
                        }
                    })

            payload = Payload(
                content=text_content if text_content else None,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                tool_calls=tool_calls_list if tool_calls_list else None,
                metadata={
                    "model": model,
                    "stop_reason": response.stop_reason,
                    "provider": "anthropic"
                }
            )

            logger.debug(f"Anthropic response: {payload.total_tokens} tokens")
            return payload

        except Exception as e:
            logger.error(f"Anthropic API error: {str(e)}")
            raise

    def list_available_models(self) -> List[str]:
        """List available Anthropic models"""
        try:
            models = self.client.models.list()
            return [model.id for model in models.data]
        except Exception as e:
            logger.warning(f"Failed to list models: {str(e)}, using defaults")
            # Return known models as fallback
            return [
                "claude-3-5-sonnet-20241022",
                "claude-3-5-haiku-20241022",
                "claude-3-opus-20240229",
                "claude-3-sonnet-20240229",
                "claude-3-haiku-20240307"
            ]

    def validate_model(self, model: str) -> bool:
        """Check if a model is available"""
        available = self.list_available_models()
        return model in available

    def format_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Format tools for Anthropic API.

        Anthropic expects tools in this format:
        {
            "name": "...",
            "description": "...",
            "input_schema": {...}
        }
        """
        # Tools should already be in the correct format from Instrument.to_anthropic_schema()
        return tools

    def supports_streaming(self) -> bool:
        return True

    def supports_json_mode(self) -> bool:
        return False  # Anthropic doesn't have native JSON mode, but can follow instructions

    def supports_vision(self) -> bool:
        return True
