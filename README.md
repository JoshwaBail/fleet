# ⛵ Fleet

**A lightweight, nautical-themed LLM agent builder for Python**

Fleet makes it easy to build AI agents that work seamlessly across multiple LLM providers (OpenAI, Anthropic, OpenRouter) with a clean, intuitive API.

## 🚀 Features

- **🎯 Simple & Lightweight** - Minimal boilerplate, maximum productivity
- **🔌 Multi-Provider** - Works with OpenAI, Anthropic, and OpenRouter out of the box
- **🛠️ Tool Support** - Easy function calling with Instruments and Arsenals
- **👥 Multi-Agent** - Coordinate multiple agents with Armada (sequential or parallel)
- **⛵ Nautical Theme** - Clean, memorable API with seafaring metaphors
- **🎨 Type-Safe** - Full type hints for better IDE support

## 📦 Installation

```bash
pip install fleet
```

Or install from source:

```bash
git clone https://github.com/yourusername/fleet.git
cd fleet
pip install -e .
```

## 🏁 Quick Start

```python
from fleet import ChatCaptain, OpenAIProvider
import openai

# Initialize provider
provider = OpenAIProvider(openai.Client(api_key="your-api-key"))

# Create a captain (agent)
captain = ChatCaptain(
    provider=provider,
    name="Navigator",
    system_prompt="You are a helpful AI assistant.",
    default_model="gpt-4o-mini"
)

# Chat!
response = captain.chat("What's the weather like today?")
print(response.content)
```

## 🗺️ Core Concepts

Fleet uses nautical/spacefaring terminology to make the API memorable and fun:

| Concept | Description |
|---------|-------------|
| **Captain** | An AI agent that can chat and/or use tools |
| **ChatCaptain** | Simple conversational agent (no tools) |
| **ToolCaptain** | Agent with function calling capabilities |
| **Instrument** | A tool/function that a captain can use |
| **Arsenal** | A collection of instruments (toolbox) |
| **Armada** | Multiple captains working together |
| **Voyage** | A mission or task |
| **Payload** | Response data from a captain |
| **Provider** | LLM API provider (OpenAI, Anthropic, etc.) |

## 📚 Usage Examples

### Simple Chat

```python
from fleet import ChatCaptain, OpenAIProvider
import openai

provider = OpenAIProvider(openai.Client(api_key="..."))

captain = ChatCaptain(
    provider=provider,
    name="Assistant",
    default_model="gpt-4o-mini"
)

# Single message
response = captain.chat("Explain quantum computing in one sentence")
print(response.content)

# Conversation (maintains context)
response1 = captain.chat("What is Python?")
response2 = captain.chat("What are its main use cases?")  # Remembers context
```

### Using Tools (Instruments)

```python
from fleet import ToolCaptain, OpenAIProvider, instrument, Arsenal
import openai

# Define tools using decorator
@instrument(
    name="get_weather",
    description="Get current weather for a location"
)
def get_weather(location: str, units: str = "celsius") -> dict:
    # Your weather API call here
    return {"temp": 22, "condition": "sunny"}

@instrument(
    name="calculate",
    description="Perform mathematical calculations"
)
def calculate(expression: str) -> float:
    return eval(expression)  # Use safely in production!

# Create arsenal
arsenal = Arsenal("Utility Tools")
arsenal.add_instrument(get_weather)
arsenal.add_instrument(calculate)

# Create captain with tools
provider = OpenAIProvider(openai.Client(api_key="..."))
captain = ToolCaptain(
    provider=provider,
    name="Tool User",
    arsenal=arsenal,
    default_model="gpt-4o-mini"
)

# Captain will automatically use tools as needed
response = captain.chat("What's the weather in Paris? Also, what's 15 * 23?")
print(response.content)
```

### Multiple Providers

```python
from fleet import ChatCaptain, OpenAIProvider, AnthropicProvider, OpenRouterProvider
import openai
import anthropic

# OpenAI
openai_provider = OpenAIProvider(openai.Client(api_key="..."))
openai_captain = ChatCaptain(provider=openai_provider, default_model="gpt-4o")

# Anthropic (Claude)
anthropic_provider = AnthropicProvider(anthropic.Anthropic(api_key="..."))
claude_captain = ChatCaptain(provider=anthropic_provider, default_model="claude-3-5-sonnet-20241022")

# OpenRouter (access to many models)
openrouter_provider = OpenRouterProvider(api_key="...")
openrouter_captain = ChatCaptain(provider=openrouter_provider, default_model="meta-llama/llama-3.1-70b-instruct")

# Use any captain the same way
question = "Explain AI in simple terms"
print(openai_captain.chat(question).content)
print(claude_captain.chat(question).content)
print(openrouter_captain.chat(question).content)
```

### Multi-Agent with Armada (Sequential)

```python
from fleet import ChatCaptain, Armada, OpenAIProvider
import openai

provider = OpenAIProvider(openai.Client(api_key="..."))

# Create specialized captains
researcher = ChatCaptain(
    provider=provider,
    name="Researcher",
    system_prompt="You gather and present factual information.",
    default_model="gpt-4o-mini"
)

analyst = ChatCaptain(
    provider=provider,
    name="Analyst",
    system_prompt="You analyze information and identify patterns.",
    default_model="gpt-4o-mini"
)

writer = ChatCaptain(
    provider=provider,
    name="Writer",
    system_prompt="You transform analysis into clear prose.",
    default_model="gpt-4o-mini"
)

# Create armada (sequential pipeline)
armada = Armada(
    captains=[researcher, analyst, writer],
    name="Content Pipeline",
    description="Research → Analyze → Write"
)

# Execute sequential voyage
result = armada.voyage(
    message="Topic: The future of AI",
    model="gpt-4o-mini",
    mode="sequential"
)

print(result.content)
```

### Multi-Agent with Armada (Parallel)

```python
from fleet import ChatCaptain, Armada, OpenAIProvider
import openai

provider = OpenAIProvider(openai.Client(api_key="..."))

# Create captains with different perspectives
tech_expert = ChatCaptain(provider=provider, name="Tech",
                         system_prompt="Focus on technical aspects")
business_expert = ChatCaptain(provider=provider, name="Business",
                             system_prompt="Focus on business value")
ux_expert = ChatCaptain(provider=provider, name="UX",
                       system_prompt="Focus on user experience")

# Create armada with parallel execution
armada = Armada(
    captains=[tech_expert, business_expert, ux_expert],
    name="Product Review Team",
    synthesize=True  # Combine perspectives into one answer
)

# Execute parallel voyage
result = armada.voyage(
    message="Should we build a mobile app?",
    model="gpt-4o-mini",
    mode="parallel"
)

print(result.content)  # Synthesized answer from all perspectives
```

## 🏗️ Architecture

```
fleet/
├── captains/           # Agent implementations
│   ├── base_captain.py    # Base agent class
│   ├── chat_captain.py    # Simple chat agent
│   └── tool_captain.py    # Agent with function calling
├── armada/             # Multi-agent orchestration
│   └── armada.py          # Armada coordinator
├── instruments/        # Tool building system
│   ├── instrument.py      # Tool/function wrapper
│   └── arsenal.py         # Tool collection manager
├── providers/          # LLM provider abstractions
│   ├── base_provider.py      # Provider interface
│   ├── openai_provider.py    # OpenAI implementation
│   ├── anthropic_provider.py # Anthropic implementation
│   └── openrouter_provider.py # OpenRouter implementation
└── payload/            # Response handling
    └── payload.py         # Response data class
```

## 🛠️ Advanced Features

### Custom Instruments with Parameters

```python
from fleet import Instrument, InstrumentParameter

search_tool = Instrument(
    name="search",
    description="Search the web",
    function=your_search_function,
    parameters=[
        InstrumentParameter("query", "string", "Search query", required=True),
        InstrumentParameter("limit", "integer", "Max results", required=False, default=10)
    ]
)
```

### Arsenal Management

```python
from fleet import Arsenal, create_arsenal

# Method 1: Direct creation
arsenal = Arsenal("My Tools")
arsenal.add_instrument(tool1)
arsenal.add_instrument(tool2)

# Method 2: Builder pattern
arsenal = create_arsenal("My Tools", "Description")\
    .with_instrument(tool1)\
    .with_instrument(tool2)\
    .build()

# Merge arsenals
combined = arsenal1.merge(arsenal2)
```

### Conversation Management

```python
captain = ChatCaptain(provider=provider, default_model="gpt-4o-mini")

# Send messages
captain.chat("Hello")
captain.chat("How are you?")

# Check history
messages = captain.get_messages()
print(f"Total messages: {len(messages)}")

# Clear history
captain.clear_messages()
```

### Token Usage and Cost Tracking

```python
response = captain.chat("Tell me a story")

print(f"Input tokens: {response.input_tokens}")
print(f"Output tokens: {response.output_tokens}")
print(f"Total tokens: {response.total_tokens}")
print(f"Estimated cost: ${response.cost_estimate:.4f}")
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

Built with inspiration from the nautical tradition of exploration and discovery.

---

**Set sail with Fleet!** ⛵
