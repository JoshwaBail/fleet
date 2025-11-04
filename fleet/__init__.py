"""
Fleet - A lightweight, nautical-themed LLM agent builder.

Fleet makes it easy to build AI agents that work across multiple providers
(OpenAI, Anthropic, OpenRouter) with a clean, intuitive API.

Quick Start:
    from fleet import ChatCaptain, create_provider

    # Cookie-cutter provider creation
    provider = create_provider("openai", api_key="your-key")
    captain = ChatCaptain(provider, name="MyBot", default_model="gpt-4o-mini")
    response = captain.chat("Hello!")
    print(response.content)
"""

__version__ = "0.2.0"

# Core Captains (Agents)
from fleet.captains import (
    BaseCaptain,
    ChatCaptain,
    ToolCaptain,
    # Advanced Patterns
    Navigator,
    Quartermaster,
    Admiral,
    WatchChange
)

# Multi-agent orchestration
from fleet.armada import (
    Armada,
    # Advanced Patterns
    HarborMaster,
    FleetCommand,
    Council
)

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
    OpenRouterProvider,
    create_provider,
    openai_provider,
    anthropic_provider,
    openrouter_provider
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
    "Navigator",  # ReAct pattern
    "Quartermaster",  # Reflection pattern
    "Admiral",  # Plan-and-Execute pattern
    "WatchChange",  # Handoff pattern

    # Armada
    "Armada",
    "HarborMaster",  # Router/Triage pattern
    "FleetCommand",  # Hierarchical pattern
    "Council",  # Debate pattern


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
    "create_provider",
    "openai_provider",
    "anthropic_provider",
    "openrouter_provider",

    # Payloads
    "Payload",
    "StreamingPayload",
]
