"""
Fleet - A lightweight, nautical-themed LLM agent builder.

Fleet makes it easy to build AI agents that work across multiple providers
(OpenAI, Anthropic, OpenRouter) with a clean, intuitive API.

Quick Start:
    from fleet import ChatCaptain, OpenAIProvider
    import openai

    provider = OpenAIProvider(openai.Client(api_key="your-key"))
    captain = ChatCaptain(provider, name="MyBot")
    response = captain.chat("Hello!", model="gpt-4")
    print(response.content)
"""

__version__ = "0.2.0"

# Core Captains (Agents)
from fleet.captains import (
    BaseCaptain,
    ChatCaptain,
    ToolCaptain
)

# Multi-agent orchestration
from fleet.armada import Armada

# Instruments (Tools)
from fleet.instruments import (
    Instrument,
    InstrumentParameter,
    instrument,
    build_instrument,
    Arsenal,
    ArsenalBuilder,
    create_arsenal
)

# Providers
from fleet.providers import (
    BaseProvider,
    OpenAIProvider,
    AnthropicProvider,
    OpenRouterProvider
)

# Payloads (Responses)
from fleet.payload import (
    Payload,
    StreamingPayload
)

__all__ = [
    # Version
    "__version__",

    # Captains
    "BaseCaptain",
    "ChatCaptain",
    "ToolCaptain",

    # Armada
    "Armada",

    # Instruments
    "Instrument",
    "InstrumentParameter",
    "instrument",
    "build_instrument",
    "Arsenal",
    "ArsenalBuilder",
    "create_arsenal",

    # Providers
    "BaseProvider",
    "OpenAIProvider",
    "AnthropicProvider",
    "OpenRouterProvider",

    # Payloads
    "Payload",
    "StreamingPayload",
]
