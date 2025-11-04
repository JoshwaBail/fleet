"""
OpenRouter Provider - Implementation for OpenRouter API.

OpenRouter provides access to multiple LLM providers through a unified API.
It's compatible with OpenAI's API format.
"""

from typing import List, Dict, Any, Optional
from fleet.providers.base_provider import BaseProvider
from fleet.payload.payload import Payload
import openai
import json
import logging

logger = logging.getLogger(__name__)


class OpenRouterProvider(BaseProvider):
    """
    Provider implementation for OpenRouter API.

    OpenRouter uses OpenAI-compatible endpoints, so we use the OpenAI client
    but with OpenRouter-specific configuration.
    """

    def __init__(self, api_key: str, site_url: Optional[str] = None, site_name: Optional[str] = None):
        """
        Initialize OpenRouter provider.

        Args:
            api_key: OpenRouter API key
            site_url: Your site URL (for rankings)
            site_name: Your site name (for rankings)
        """
        # Create OpenAI client configured for OpenRouter
        client = openai.Client(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1"
        )

        super().__init__(client, "OpenRouter")

        self.site_url = site_url
        self.site_name = site_name
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
        """Send a message using OpenRouter API"""

        # Prepare request parameters
        request_params = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs
        }

        # Add OpenRouter-specific headers if provided
        extra_headers = {}
        if self.site_url:
            extra_headers["HTTP-Referer"] = self.site_url
        if self.site_name:
            extra_headers["X-Title"] = self.site_name

        if extra_headers:
            request_params["extra_headers"] = extra_headers

        # Add tools if provided
        if tools:
            request_params["tools"] = tools
            request_params["tool_choice"] = tool_choice

        logger.debug(f"OpenRouter request: model={model}, messages={len(messages)}, tools={len(tools) if tools else 0}")

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

            # OpenRouter may not always provide usage stats
            input_tokens = getattr(response.usage, 'prompt_tokens', 0) if response.usage else 0
            output_tokens = getattr(response.usage, 'completion_tokens', 0) if response.usage else 0

            payload = Payload(
                content=content,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                tool_calls=tool_calls_list,
                metadata={
                    "model": model,
                    "finish_reason": response.choices[0].finish_reason,
                    "provider": "openrouter"
                }
            )

            logger.debug(f"OpenRouter response: {payload.total_tokens} tokens")
            return payload

        except Exception as e:
            logger.error(f"OpenRouter API error: {str(e)}")
            raise

    def list_available_models(self) -> List[str]:
        """
        List available OpenRouter models.

        Note: OpenRouter has many models, so this might be a large list.
        """
        if self._cached_models is None:
            try:
                # OpenRouter provides a models endpoint
                import requests
                response = requests.get("https://openrouter.ai/api/v1/models")
                if response.status_code == 200:
                    data = response.json()
                    self._cached_models = [model["id"] for model in data.get("data", [])]
                else:
                    logger.warning(f"Failed to fetch models: {response.status_code}")
                    self._cached_models = self._get_popular_models()
            except Exception as e:
                logger.error(f"Failed to list models: {str(e)}")
                self._cached_models = self._get_popular_models()

        return self._cached_models

    def _get_popular_models(self) -> List[str]:
        """Return a list of popular OpenRouter models as fallback"""
        return [
            # Anthropic
            "anthropic/claude-3.5-sonnet",
            "anthropic/claude-3-opus",
            "anthropic/claude-3-sonnet",
            "anthropic/claude-3-haiku",
            # OpenAI
            "openai/gpt-4o",
            "openai/gpt-4-turbo",
            "openai/gpt-4",
            "openai/gpt-3.5-turbo",
            # Google
            "google/gemini-pro-1.5",
            "google/gemini-pro",
            # Meta
            "meta-llama/llama-3.1-405b-instruct",
            "meta-llama/llama-3.1-70b-instruct",
            "meta-llama/llama-3.1-8b-instruct",
            # Mistral
            "mistralai/mistral-large",
            "mistralai/mixtral-8x7b-instruct",
            # Others
            "cohere/command-r-plus",
            "qwen/qwen-2-72b-instruct",
        ]

    def validate_model(self, model: str) -> bool:
        """Check if a model is available"""
        available = self.list_available_models()
        return model in available

    def format_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Format tools for OpenRouter API.

        OpenRouter uses OpenAI-compatible format.
        """
        return tools

    def supports_streaming(self) -> bool:
        return True

    def supports_json_mode(self) -> bool:
        return True  # Many models support it through OpenRouter

    def supports_vision(self) -> bool:
        return True  # Many models support vision through OpenRouter
