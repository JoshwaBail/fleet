"""
Provider Factory - Easy provider creation with uniform interface.

This factory makes it trivial to switch between providers using
a consistent API.
"""

from typing import Optional, Any
from fleet.providers.base_provider import BaseProvider
from fleet.providers.openai_provider import OpenAIProvider
from fleet.providers.anthropic_provider import AnthropicProvider
from fleet.providers.openrouter_provider import OpenRouterProvider
import logging

logger = logging.getLogger(__name__)


def create_provider(
    provider_type: str,
    api_key: Optional[str] = None,
    client: Optional[Any] = None,
    **kwargs
) -> BaseProvider:
    """
    Factory function to create a provider.

    This provides a uniform interface for creating any provider type.
    Perfect for dependency injection and easy provider switching.

    Args:
        provider_type: Type of provider ("openai", "anthropic", "openrouter")
        api_key: API key for the provider
        client: Pre-configured client (optional)
        **kwargs: Additional provider-specific arguments

    Returns:
        Configured provider instance

    Examples:
        # Cookie-cutter usage - just change the provider_type!
        provider = create_provider("openai", api_key="sk-...")
        provider = create_provider("anthropic", api_key="sk-ant-...")
        provider = create_provider("openrouter", api_key="sk-or-...")

        # All providers work exactly the same
        captain = ChatCaptain(provider=provider, ...)
    """
    provider_type = provider_type.lower()

    if provider_type in ["openai", "gpt"]:
        return OpenAIProvider(api_key=api_key, client=client, **kwargs)

    elif provider_type in ["anthropic", "claude"]:
        return AnthropicProvider(api_key=api_key, client=client, **kwargs)

    elif provider_type in ["openrouter", "or"]:
        return OpenRouterProvider(api_key=api_key, client=client, **kwargs)

    else:
        raise ValueError(
            f"Unknown provider type: {provider_type}. "
            f"Supported: 'openai', 'anthropic', 'openrouter'"
        )


# Convenience aliases
def openai_provider(api_key: Optional[str] = None, **kwargs) -> OpenAIProvider:
    """Create an OpenAI provider"""
    return OpenAIProvider(api_key=api_key, **kwargs)


def anthropic_provider(api_key: Optional[str] = None, **kwargs) -> AnthropicProvider:
    """Create an Anthropic provider"""
    return AnthropicProvider(api_key=api_key, **kwargs)


def openrouter_provider(api_key: Optional[str] = None, **kwargs) -> OpenRouterProvider:
    """Create an OpenRouter provider"""
    return OpenRouterProvider(api_key=api_key, **kwargs)
