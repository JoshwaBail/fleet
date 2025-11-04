# Fleet Examples: Pattern Selection Guide

This directory contains examples demonstrating different agent patterns and orchestration strategies in Fleet. Each pattern solves specific types of problems, and choosing the right one can dramatically improve your application's performance, cost, and user experience.

## Quick Navigation

- [Pattern Selection Guide](#pattern-selection-guide) - Start here to find the right pattern
- [Examples by Pattern](#examples-by-pattern) - Browse all examples
- [Use Case Mappings](#use-case-mappings) - Find patterns by industry/task
- [Pattern Comparison](#pattern-comparison-matrix) - Compare patterns side-by-side
- [Getting Started](#getting-started) - First-time user guide

---

## Pattern Selection Guide

### Decision Tree

```
Do you need multiple agents?
│
├─ NO → Use ChatCaptain or ToolCaptain
│   │
│   ├─ Need tools? → ToolCaptain (Example 02)
│   └─ Just chat? → ChatCaptain (Example 01)
│
└─ YES → Continue...
    │
    Do you need quality iteration?
    │
    ├─ YES →
    │   ├─ Single agent improving? → Quartermaster (Reflection, Example 08)
    │   ├─ Multi-agent debate? → Council (Debate, Example 11)
    │   └─ Step-by-step reasoning? → Navigator (ReAct, Example 07)
    │
    └─ NO → Continue...
        │
        Do you have specialized agents?
        │
        ├─ YES →
        │   ├─ Route to one? → HarborMaster (Router, Example 09)
        │   ├─ Director + workers? → FleetCommand (Hierarchical, Example 10)
        │   └─ Pass between agents? → WatchChange (Handoff, Example 12)
        │
        └─ NO → Simple composition
            ├─ Sequential pipeline? → Armada Sequential (Example 04)
            └─ Parallel + synthesis? → Armada Parallel (Example 05)
```

### Quick Pattern Reference

| Pattern | When to Use | Example |
|---------|-------------|---------|
| **ChatCaptain** | Simple conversations, no tools needed | 01 |
| **ToolCaptain** | Need function calling capabilities | 02 |
| **Navigator** (ReAct) | Complex reasoning with tools, show your work | 07 |
| **Quartermaster** (Reflection) | Quality matters more than speed, iterative improvement | 08 |
| **Admiral** (Plan-Execute) | Complex tasks benefit from upfront planning | 13 |
| **HarborMaster** (Router) | Route requests to specialized agents | 09 |
| **WatchChange** (Handoff) | Context-preserving agent transfers | 12 |
| **FleetCommand** (Hierarchical) | Director coordinates specialized workers | 10 |
| **Council** (Debate) | Multiple perspectives needed, consensus building | 11 |
| **Armada Sequential** | Multi-step pipeline workflow | 04 |
| **Armada Parallel** | Independent tasks executed simultaneously | 05 |

---

## Examples by Pattern

### Foundation Patterns

#### 00: Cookie-Cutter Providers
**File**: `00_cookie_cutter_providers.py`

**Pattern**: Dependency Injection

**What it demonstrates**:
- True provider interchangeability
- Factory pattern for provider creation
- Switching between OpenAI, Anthropic, OpenRouter with one line
- Configuration-driven provider selection

**Use cases**:
- Multi-tenant applications
- A/B testing different models
- Cost optimization by provider
- Fallback strategies

```python
# One line to switch providers!
provider = create_provider("openai", api_key="...")
# provider = create_provider("anthropic", api_key="...")
# provider = create_provider("openrouter", api_key="...")
```

---

#### 01: Basic Chat
**File**: `01_basic_chat.py`

**Pattern**: Single Agent Conversation

**What it demonstrates**:
- Simple ChatCaptain usage
- Conversation memory
- Message history management

**Use cases**:
- Chatbots
- Conversational interfaces
- Simple Q&A systems

---

#### 02: Tool-Using Agent
**File**: `02_tool_captain.py`

**Pattern**: Single Agent with Tools

**What it demonstrates**:
- ToolCaptain with function calling
- Creating instruments
- Tool execution and result handling

**Use cases**:
- Agents that need to access external data
- API integrations
- Database queries
- File operations

---

#### 03: Instrument Builder
**File**: `03_instrument_builder.py`

**Pattern**: Tool Creation

**What it demonstrates**:
- @instrument decorator
- Automatic parameter inference
- Cross-provider schema generation
- Arsenal (toolbox) management

**Use cases**:
- Building reusable tool libraries
- Custom function collections
- Domain-specific toolsets

---

### Composition Patterns

#### 04: Sequential Pipeline
**File**: `04_armada_sequential.py`

**Pattern**: Sequential Composition

**What it demonstrates**:
- Armada in sequential mode
- Pipeline workflows
- Output chaining between agents

**Use cases**:
- Multi-stage processing (research → analysis → summary)
- Data transformation pipelines
- Progressive refinement workflows

**Characteristics**:
- ⏱️ Slower (sequential execution)
- 💰 Lower cost (fewer total tokens)
- 🎯 High coherence (each step builds on previous)

---

#### 05: Parallel Execution
**File**: `05_armada_parallel.py`

**Pattern**: Parallel Composition

**What it demonstrates**:
- Armada in parallel mode
- Concurrent agent execution
- Result synthesis

**Use cases**:
- Independent analyses that need synthesis
- Multiple perspectives on same input
- Consensus building
- Redundancy for reliability

**Characteristics**:
- ⚡ Faster (parallel execution)
- 💰 Higher cost (more total tokens)
- 🎯 Diverse perspectives

---

### Advanced Reasoning Patterns

#### 07: Navigator (ReAct)
**File**: `07_navigator_react.py`

**Pattern**: Reasoning and Acting (ReAct)

**What it demonstrates**:
- Think → Act → Observe iteration
- Visible reasoning trace
- Tool-augmented problem solving
- Step-by-step transparency

**Use cases**:
- Complex data analysis
- Multi-step research tasks
- Debugging and troubleshooting
- Tasks requiring explanation of reasoning

**Characteristics**:
- ⏱️ Moderate speed (iterative but focused)
- 💰 Moderate cost (multiple reasoning rounds)
- 🎯 Excellent transparency
- 🧠 Strong for complex reasoning

```python
navigator = Navigator(
    provider=provider,
    instruments=[calculator, search, database],
    max_iterations=10,
    verbose=True  # Shows 💭 Thought → 🔧 Action → 👁️ Observation
)
```

**When to use**:
- You need to explain HOW the answer was reached
- Task requires multiple steps with tools
- Debugging complex problems
- Teaching/demonstration scenarios

**When NOT to use**:
- Simple single-step tasks
- Speed is critical
- No tools needed

---

#### 08: Quartermaster (Reflection)
**File**: `08_quartermaster_reflection.py`

**Pattern**: Generate-Critique-Refine (Reflection)

**What it demonstrates**:
- Iterative quality improvement
- Self-critique mechanism
- Version history tracking
- Content refinement

**Use cases**:
- High-quality content creation
- Technical writing
- Code review and improvement
- Documentation generation

**Characteristics**:
- ⏱️ Slower (multiple iterations)
- 💰 Higher cost (generation + critique cycles)
- 🎯 Highest quality output
- 📈 Measurable improvement over iterations

```python
quartermaster = Quartermaster(
    provider=provider,
    critique_prompt="Review for: accuracy, clarity, completeness...",
    max_iterations=3,
    auto_improve=True
)
```

**When to use**:
- Quality is more important than speed
- Content needs refinement
- Iterative improvement desired
- You want to see improvement history

**When NOT to use**:
- Real-time responses needed
- First draft is sufficient
- Cost is primary concern

---

#### 13: Admiral (Plan-and-Execute)
**File**: `13_admiral_plan_execute.py` (if created)

**Pattern**: Planning then Execution

**What it demonstrates**:
- Separate planning phase
- Step-by-step execution
- Different models for planning vs execution
- Replanning on failures

**Use cases**:
- Complex multi-step projects
- Resource-constrained environments
- Tasks benefiting from upfront planning
- Cost optimization (cheap planner, expensive executor)

**Characteristics**:
- ⏱️ Moderate speed (planning overhead)
- 💰 Flexible cost (can use different model tiers)
- 🎯 Systematic approach
- 🔄 Supports replanning

**When to use**:
- Complex tasks with many steps
- Want to review plan before execution
- Different models for different phases
- Execution might fail and need replanning

**When NOT to use**:
- Simple tasks
- Highly dynamic environments
- Planning overhead not justified

---

### Coordination Patterns

#### 09: HarborMaster (Router)
**File**: `09_harbor_master_router.py`

**Pattern**: Intelligent Routing

**What it demonstrates**:
- Request routing to specialists
- LLM-based or rules-based routing
- Routing statistics
- Specialist management

**Use cases**:
- Customer service triage
- Multi-domain applications
- Specialized expert systems
- Request classification

**Characteristics**:
- ⚡ Fast (single routing decision)
- 💰 Low cost (one router + one specialist)
- 🎯 Efficient specialization
- 📊 Trackable routing patterns

```python
harbor_master = HarborMaster(
    specialists={
        "technical": tech_agent,
        "billing": billing_agent,
        "sales": sales_agent
    },
    routing_strategy="llm"  # or "rules"
)
```

**When to use**:
- Clear specialist domains
- Request classification possible
- Want efficiency over consensus
- Track routing patterns

**When NOT to use**:
- Single domain
- All requests need same handling
- Multiple perspectives required

---

#### 10: FleetCommand (Hierarchical)
**File**: `10_fleet_command_hierarchical.py`

**Pattern**: Director + Workers

**What it demonstrates**:
- Hierarchical orchestration
- Task decomposition
- Parallel worker execution
- Result synthesis

**Use cases**:
- Complex project coordination
- Research with multiple angles
- System design (research + architect + writer)
- Multi-faceted analysis

**Characteristics**:
- ⏱️ Moderate speed (parallel workers but 3 phases)
- 💰 Moderate-high cost (director + all workers)
- 🎯 Comprehensive results
- 🏗️ Structured approach

```python
fleet_command = FleetCommand(
    director=project_director,
    workers=[researcher, architect, writer],
    task_decomposition_strategy="llm",
    synthesize_results=True
)
```

**When to use**:
- Complex tasks needing decomposition
- Multiple specialized roles needed
- Want structured coordination
- Comprehensive deliverables

**When NOT to use**:
- Simple tasks
- No clear specializations
- Cost is primary concern
- Single perspective sufficient

---

#### 11: Council (Debate)
**File**: `11_council_debate.py` (if created)

**Pattern**: Multi-Agent Debate

**What it demonstrates**:
- Multiple perspectives
- Debate rounds with rebuttals
- Resolution strategies (synthesis/vote/consensus)
- Argumentation tracking

**Use cases**:
- Decision making with uncertainty
- Evaluating trade-offs
- Red team / blue team scenarios
- Consensus building

**Characteristics**:
- ⏱️ Slow (multiple rounds × multiple agents)
- 💰 Highest cost (many agents × many rounds)
- 🎯 Diverse perspectives
- 🤝 Consensus or synthesized decisions

**When to use**:
- Important decisions
- Multiple valid perspectives
- Need to explore trade-offs
- Consensus desired

**When NOT to use**:
- Clear correct answer
- Speed critical
- Cost-sensitive
- Single perspective sufficient

---

#### 12: WatchChange (Handoff)
**File**: `12_watch_change_handoff.py` (if created)

**Pattern**: Context-Preserving Handoff

**What it demonstrates**:
- Agent-to-agent handoffs
- Context preservation
- Handoff chain tracking
- Max handoff limits

**Use cases**:
- Customer service escalation
- Specialist consultation
- Progressive specialization
- Transfer workflows

**Characteristics**:
- ⏱️ Variable (depends on handoff chain)
- 💰 Moderate (only active agents pay)
- 🎯 Smooth transitions
- 🔄 Trackable handoff history

```python
# Agent can hand off to another
support_agent = WatchChange(...)
specialist_agent = WatchChange(...)
support_agent.register_handoff("escalate", specialist_agent)
```

**When to use**:
- Clear handoff conditions
- Specialist consultation needed
- Progressive complexity
- Context must be preserved

**When NOT to use**:
- No clear handoff criteria
- Single agent sufficient
- Handoff overhead not justified

---

## Use Case Mappings

### By Industry

#### Customer Service
- **Triage**: HarborMaster (09) - Route to right specialist
- **Escalation**: WatchChange (12) - Hand off complex cases
- **FAQ**: ChatCaptain (01) - Simple Q&A

#### Content Creation
- **High quality**: Quartermaster (08) - Iterate for quality
- **Multiple perspectives**: Council (11) - Debate approaches
- **Technical docs**: FleetCommand (10) - Research + architecture + writing

#### Software Development
- **Code review**: Quartermaster (08) - Iterative improvement
- **System design**: FleetCommand (10) - Multiple specialists
- **Debugging**: Navigator (07) - Reason through problem
- **Planning**: Admiral (13) - Plan then implement

#### Research & Analysis
- **Multi-source**: Armada Parallel (05) - Concurrent research
- **Deep analysis**: Navigator (07) - Reason with tools
- **Comprehensive**: FleetCommand (10) - Director + specialist workers

#### Data Analysis
- **Complex queries**: Navigator (07) - Reason through data
- **Multiple datasets**: Armada Parallel (05) - Parallel analysis
- **Report generation**: Armada Sequential (04) - Query → analyze → summarize

### By Task Type

#### Decision Making
1. **Simple**: ChatCaptain (01)
2. **With data**: ToolCaptain (02) or Navigator (07)
3. **Multiple perspectives**: Council (11)
4. **Specialist input**: HarborMaster (09) or FleetCommand (10)

#### Content Generation
1. **First draft**: ChatCaptain (01)
2. **High quality**: Quartermaster (08)
3. **Multiple sections**: Armada Sequential (04)
4. **Multiple perspectives**: Armada Parallel (05)

#### Problem Solving
1. **Simple**: ToolCaptain (02)
2. **Complex reasoning**: Navigator (07)
3. **Planning needed**: Admiral (13)
4. **Specialist help**: WatchChange (12) or HarborMaster (09)

#### Coordination
1. **Pipeline**: Armada Sequential (04)
2. **Parallel**: Armada Parallel (05)
3. **Routing**: HarborMaster (09)
4. **Hierarchical**: FleetCommand (10)
5. **Handoff**: WatchChange (12)

---

## Pattern Comparison Matrix

| Pattern | Speed | Cost | Quality | Complexity | Transparency | Best For |
|---------|-------|------|---------|------------|--------------|----------|
| ChatCaptain | ⚡⚡⚡ | 💰 | ⭐⭐ | Simple | Low | Chat, Q&A |
| ToolCaptain | ⚡⚡⚡ | 💰💰 | ⭐⭐⭐ | Simple | Medium | Single-agent tools |
| Navigator | ⚡⚡ | 💰💰 | ⭐⭐⭐⭐ | Moderate | High | Reasoning with tools |
| Quartermaster | ⚡ | 💰💰💰 | ⭐⭐⭐⭐⭐ | Moderate | High | Quality content |
| Admiral | ⚡⚡ | 💰💰 | ⭐⭐⭐⭐ | Moderate | High | Complex tasks |
| HarborMaster | ⚡⚡⚡ | 💰💰 | ⭐⭐⭐ | Moderate | Medium | Routing/Triage |
| WatchChange | ⚡⚡ | 💰💰 | ⭐⭐⭐ | Moderate | Medium | Escalation |
| FleetCommand | ⚡⚡ | 💰💰💰 | ⭐⭐⭐⭐ | Complex | Medium | Project coordination |
| Council | ⚡ | 💰💰💰💰 | ⭐⭐⭐⭐⭐ | Complex | High | Consensus/Debate |
| Sequential | ⚡ | 💰💰 | ⭐⭐⭐⭐ | Simple | Medium | Pipelines |
| Parallel | ⚡⚡ | 💰💰💰 | ⭐⭐⭐ | Moderate | Medium | Multiple perspectives |

### Legend
- Speed: ⚡⚡⚡ (Fast) → ⚡ (Slow)
- Cost: 💰 (Cheap) → 💰💰💰💰 (Expensive)
- Quality: ⭐⭐ (Basic) → ⭐⭐⭐⭐⭐ (Excellent)

---

## Getting Started

### For First-Time Users

1. **Start with basics** (Examples 00-03)
   - Understand provider interchangeability (00)
   - Learn simple chat (01)
   - Add tools (02, 03)

2. **Learn composition** (Examples 04-05)
   - Sequential pipelines (04)
   - Parallel execution (05)

3. **Try advanced patterns** (Examples 07-10)
   - Start with Navigator (07) - easiest to understand
   - Try Quartermaster (08) for quality improvement
   - Experiment with coordination (09, 10)

### Choosing Your First Pattern

Ask yourself:
- **Do I need multiple agents?** → No? Start with ChatCaptain/ToolCaptain
- **Do I need quality iteration?** → Yes? Try Quartermaster
- **Do I have specialized agents?** → Yes? Try HarborMaster or FleetCommand
- **Do I need complex reasoning?** → Yes? Try Navigator

### Common Combinations

Patterns can be combined:

```python
# Example: Router + Reflection
harbor_master = HarborMaster(
    specialists={
        "writer": Quartermaster(...),  # Reflection for quality
        "analyst": Navigator(...),      # ReAct for reasoning
    }
)
```

---

## Learning Path

### Beginner
1. Example 00: Cookie-Cutter Providers
2. Example 01: Basic Chat
3. Example 02: Tool Captain
4. Example 03: Instrument Builder

**Goal**: Understand single agents and tools

### Intermediate
5. Example 04: Sequential Armada
6. Example 05: Parallel Armada
7. Example 07: Navigator (ReAct)
8. Example 08: Quartermaster (Reflection)

**Goal**: Understand composition and reasoning patterns

### Advanced
9. Example 09: HarborMaster (Router)
10. Example 10: FleetCommand (Hierarchical)
11. Example 11: Council (Debate)
12. Example 12: WatchChange (Handoff)
13. Example 13: Admiral (Plan-Execute)

**Goal**: Master complex coordination patterns

---

## Pattern Anti-Patterns

### What NOT to do

❌ **Using Council for simple questions**
- Council is expensive and slow
- Use ChatCaptain for simple Q&A

❌ **Using Sequential when order doesn't matter**
- Sequential is slower than parallel
- Use Parallel if tasks are independent

❌ **Using Reflection for real-time responses**
- Reflection takes multiple iterations
- Use direct chat for speed

❌ **Using Hierarchical for single-domain tasks**
- FleetCommand has overhead
- Use single agent if no specialization needed

❌ **Over-engineering simple tasks**
- Start simple, add complexity only when needed
- Most tasks can be solved with ChatCaptain or ToolCaptain

---

## FAQ

### When should I use patterns vs simple agents?

**Use simple agents (ChatCaptain/ToolCaptain) when**:
- Task is straightforward
- Single perspective sufficient
- Speed and cost matter

**Use patterns when**:
- Task is complex or multi-faceted
- Quality/reasoning matters more than speed
- Multiple perspectives needed
- Clear specialist roles exist

### How do I choose between Navigator and Admiral?

**Navigator (ReAct)**: Better for exploratory tasks where the path isn't clear upfront. Iterates naturally.

**Admiral (Plan-Execute)**: Better when planning upfront is valuable, or you want to use different models for planning vs execution.

### Can I combine patterns?

Yes! Common combinations:
- Router + any pattern (route to specialized patterns)
- Sequential/Parallel + any pattern (compose patterns)
- Reflection + others (improve quality of any pattern's output)

### How do I optimize costs?

1. Start with simplest pattern that works
2. Use cheaper models (gpt-4o-mini instead of gpt-4)
3. Limit iterations (max_iterations for Quartermaster, max_rounds for Council)
4. Use Router to direct expensive patterns only where needed
5. Consider Admiral with cheap planner, expensive executor

### Which pattern is most cost-effective?

**For quality**: Quartermaster (fewer iterations than Council)
**For speed**: ChatCaptain or HarborMaster
**For complex reasoning**: Navigator (focused iterations)
**Overall**: It depends on your specific use case!

---

## Contributing Examples

Want to add an example? Follow this structure:

```python
"""
Example N: Title (Pattern Name)

Brief description of what this example demonstrates.

Use Case: Specific use case this example addresses.
"""

import os
from fleet import ...

def main():
    print("="*70)
    print("EXAMPLE TITLE")
    print("="*70)

    # Setup
    provider = create_provider("openai", api_key=os.getenv("OPENAI_API_KEY"))

    # Pattern implementation
    # ...

    # Usage demonstration
    # ...

    # Results display
    # ...

if __name__ == "__main__":
    main()
```

---

## Additional Resources

- **Main Documentation**: ../README.md
- **Agent Research**: ../docs/AGENT_ORCHESTRATION_RESEARCH.md
- **Pattern Comparisons**: ../docs/PATTERN_COMPARISON_MATRIX.md
- **API Reference**: (Coming soon)

---

## Quick Reference Card

```
PATTERN CHEAT SHEET

Single Agent:
  ChatCaptain → Chat without tools
  ToolCaptain → Chat with tools

Reasoning:
  Navigator → Think-Act-Observe loop
  Quartermaster → Generate-Critique-Refine
  Admiral → Plan then execute

Coordination:
  HarborMaster → Route to specialists
  FleetCommand → Director + workers
  WatchChange → Agent handoffs
  Council → Multi-agent debate

Composition:
  Sequential → Pipeline A → B → C
  Parallel → All at once, then synthesize
```

Happy sailing! ⚓
