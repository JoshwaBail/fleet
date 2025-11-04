# Fleet Examples

This directory contains examples showing how to use Fleet, a lightweight LLM agent builder with a nautical theme.

## Examples Overview

1. **01_simple_chat.py** - Basic chat with a single captain
   - Creating a ChatCaptain
   - Simple conversation
   - Context management

2. **02_tools_and_instruments.py** - Using tools/instruments
   - Creating instruments (tools)
   - Building an arsenal (toolbox)
   - ToolCaptain with function calling

3. **03_multi_provider.py** - Working with multiple providers
   - OpenAI, Anthropic, and OpenRouter
   - Comparing responses across providers
   - Provider abstraction

4. **04_armada_sequential.py** - Sequential multi-agent
   - Chain of thought processing
   - Pipeline of specialized agents
   - Context accumulation

5. **05_armada_parallel.py** - Parallel multi-agent
   - Simultaneous execution
   - Diverse perspectives
   - Response synthesis

6. **06_advanced_tools.py** - Advanced tool usage
   - Complex tool parameters
   - Multiple tool calls
   - Tool chaining

## Running Examples

1. Install dependencies:
```bash
pip install -e ..
```

2. Set up environment variables:
```bash
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"  # For Anthropic examples
export OPENROUTER_API_KEY="your-key"  # For OpenRouter examples
```

3. Run an example:
```bash
python 01_simple_chat.py
```

## Quick Start

```python
from fleet import ChatCaptain, OpenAIProvider
import openai

# Initialize provider
provider = OpenAIProvider(openai.Client(api_key="your-key"))

# Create captain
captain = ChatCaptain(
    provider=provider,
    name="MyBot",
    default_model="gpt-4o-mini"
)

# Chat
response = captain.chat("Hello, world!")
print(response.content)
```

## Nautical Theme

Fleet uses nautical/spacefaring terminology:

- **Captain** - An AI agent
- **Armada** - A group of captains working together
- **Instrument** - A tool/function the captain can use
- **Arsenal** - A collection of instruments
- **Voyage** - A mission or task
- **Payload** - The response/data returned

Happy sailing! ⛵
