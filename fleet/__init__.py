"""
Fleet - A lightweight LLM agent builder.

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
    ReActCaptain,
    ReflectiveCaptain,
    Admiral,
    HandoffCaptain
)

# Multi-agent orchestration
from fleet.armada import (
    AgentFleet,
    # Advanced Patterns
    Router,
    HierarchicalFleet,
    Council
)

# Tools
from fleet.tools import (
    Tool,
    ToolParameter,
    tool,
    build_tool,
    Toolbox,
    ToolboxBuilder,
    create_toolbox
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
    "ReActCaptain",  # ReAct pattern (formerly Navigator)
    "ReflectiveCaptain",  # Reflection pattern (formerly Quartermaster)
    "Admiral",  # Plan-and-Execute pattern
    "HandoffCaptain",  # Handoff pattern (formerly WatchChange)

    # Multi-Agent Orchestration
    "AgentFleet",  # Multi-agent orchestration (formerly Armada)
    "Router",  # Router/Triage pattern (formerly HarborMaster)
    "HierarchicalFleet",  # Hierarchical pattern (formerly FleetCommand)
    "Council",  # Debate pattern

    # Tools
    "Tool",
    "ToolParameter",
    "tool",
    "build_tool",
    "Toolbox",
    "ToolboxBuilder",
    "create_toolbox",

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
