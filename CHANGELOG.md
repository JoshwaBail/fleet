# Changelog

All notable changes to Fleet will be documented in this file.

## [0.2.0] - 2024-11-04

### 🚀 Major Refactor - Revival Release

This release represents a complete overhaul of Fleet with a strong nautical/spacefaring theme and modern architecture.

### Added

#### Core Architecture
- **Provider Abstraction Layer**: Clean separation between LLM providers
  - `BaseProvider`: Abstract interface for all providers
  - `OpenAIProvider`: OpenAI API implementation
  - `AnthropicProvider`: Anthropic/Claude API implementation
  - `OpenRouterProvider`: NEW - OpenRouter support for multi-model access

#### Captain System (Agents)
- **BaseCaptain**: Foundation class for all agents
- **ChatCaptain**: Simple conversational agents without tools
- **ToolCaptain**: Agents with full function calling capabilities
- Conversation history management
- Default parameter support for easier usage

#### Instruments & Arsenals (Tools)
- **Instrument**: Tool/function wrapper with schema generation
  - Decorator-based creation with `@instrument`
  - Automatic parameter inference
  - Support for OpenAI and Anthropic schemas
- **Arsenal**: Tool collection manager
  - Builder pattern support
  - Tool merging capabilities
  - Easy tool organization

#### Armada (Multi-Agent Orchestration)
- **Sequential Mode**: Chain of thought processing
  - Context accumulation across agents
  - Pipeline-style workflows
- **Parallel Mode**: Simultaneous execution
  - Diverse perspectives
  - Automatic synthesis of results
- Beautiful terminal output with colors
- Nested armada support

#### Payload System
- **Payload**: Rich response objects
  - Token tracking (input/output/total)
  - Cost estimation
  - Metadata support
  - Tool call tracking
- **StreamingPayload**: For future streaming support

#### Documentation & Examples
- Comprehensive README with examples
- 6 detailed example scripts:
  1. Simple chat
  2. Tools and instruments
  3. Multi-provider usage
  4. Sequential armada
  5. Parallel armada
  6. Advanced tool usage
- Examples README with quick start

### Changed
- Complete restructure from flat to modular architecture
- Renamed `Agent` → `Captain` for thematic consistency
- Renamed `Fleet` → `Armada` for better clarity
- Renamed `ResponseObject` → `Payload` with enhanced features
- Improved error handling and logging throughout
- Better type hints for IDE support

### Improved
- Much cleaner API surface
- Better separation of concerns
- More intuitive naming
- Easier to extend and customize
- Better documentation

### Technical Details
- Python 3.8+ required
- Added dependencies: pydantic, requests
- Maintained dependencies: openai, anthropic, termcolor
- All code follows modern Python best practices

---

## [0.1.0] - Initial Release

### Added
- Basic Agent class
- Fleet composition
- OpenAI, Anthropic, and Groq support
- Simple function calling
- Context management

---

**Note**: Version 0.2.0 is NOT backward compatible with 0.1.0 due to the extensive refactoring. Please refer to the migration guide if upgrading.
