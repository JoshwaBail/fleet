# Agent Pattern Comparison Matrix

Quick reference guide for choosing the right agent orchestration pattern for your use case.

---

## Pattern Selection Matrix

| Pattern | Complexity | Cost | Speed | Quality | Use Case |
|---------|-----------|------|-------|---------|----------|
| **Sequential** ✅ | Low | Low | Fast | Good | Pipeline workflows |
| **Parallel** ✅ | Low | Medium | Fast | Good | Multi-perspective |
| **ReAct** | Medium | Medium | Medium | Good | Exploratory tasks |
| **Plan-Execute** | Medium-High | Medium | Medium | Very Good | Complex multi-step |
| **Reflection** | Low-Medium | High | Slow | Excellent | Quality-critical |
| **Hierarchical** | High | High | Medium | Excellent | Large projects |
| **Swarm** | High | Very High | Slow | Variable | Brainstorming |
| **Debate** | Medium-High | High | Slow | Excellent | Decision-making |
| **Router** | Medium | Low | Fast | Good | Multi-domain apps |
| **Handoff** | Medium | Low | Fast | Good | Conversational AI |
| **Agent-as-Tool** | Low-Medium | Medium | Fast | Good | Hierarchical tasks |
| **Mixture of Agents** | Medium | Very High | Slow | Excellent | Critical outputs |
| **Guardrails** | Low | Low | Fast | Good | Safety/Compliance |
| **Meta-Agent** | Very High | High | Slow | Variable | Research/Adaptive |

✅ = Currently implemented in Fleet

---

## Decision Tree

```
START: What type of task do you have?

├─ Single agent sufficient?
│  ├─ YES: Need quality iteration?
│  │  ├─ YES → Use REFLECTION
│  │  └─ NO: Need tool use?
│  │     ├─ YES: Want to see reasoning?
│  │     │  ├─ YES → Use REACT
│  │     │  └─ NO → Use ToolCaptain ✅
│  │     └─ NO → Use ChatCaptain ✅
│  │
│  └─ NO: Multiple agents needed
│     ├─ Is there a clear sequence?
│     │  ├─ YES: Can plan upfront?
│     │  │  ├─ YES → Use PLAN-AND-EXECUTE
│     │  │  └─ NO → Use SEQUENTIAL ✅
│     │  │
│     │  └─ NO: Agents work simultaneously?
│     │     ├─ YES: Need different perspectives?
│     │     │  ├─ YES: Conflicting views valuable?
│     │     │  │  ├─ YES → Use DEBATE
│     │     │  │  └─ NO → Use PARALLEL ✅
│     │     │  └─ NO → Use PARALLEL ✅
│     │     │
│     │     └─ NO: Routing needed?
│     │        ├─ Routing based on request type?
│     │        │  ├─ YES → Use ROUTER
│     │        │  └─ NO: Dynamic handoffs needed?
│     │        │     ├─ YES → Use HANDOFF
│     │        │     └─ NO: Clear hierarchy?
│     │        │        ├─ YES → Use HIERARCHICAL
│     │        │        └─ NO → Use SWARM
│
└─ Special requirements?
   ├─ Need safety checks? → Add GUARDRAILS
   ├─ Want best of multiple models? → Use MIXTURE OF AGENTS
   ├─ One agent calls others as tools? → Use AGENT-AS-TOOL
   └─ Completely dynamic? → Use META-AGENT (experimental)
```

---

## Use Case → Pattern Mapping

### Customer Service / Support
1. **Primary**: Router (triage to specialists)
2. **Secondary**: Handoff (escalation)
3. **Safety**: Guardrails (compliance)

### Content Creation
1. **Primary**: Sequential (research → write → edit)
2. **Quality**: Reflection (iterative improvement)
3. **Multi-perspective**: Debate (different angles)

### Data Analysis
1. **Primary**: Plan-and-Execute (structured analysis)
2. **Exploration**: ReAct (interactive exploration)
3. **Validation**: Debate (challenge assumptions)

### Software Development
1. **Primary**: Hierarchical (architect → developers → testers)
2. **Planning**: Plan-and-Execute (feature planning)
3. **Review**: Debate (design review)

### Research & Investigation
1. **Primary**: ReAct (explore and learn)
2. **Synthesis**: Parallel → Synthesis (multiple sources)
3. **Verification**: Reflection (self-check)

### Decision Making
1. **Primary**: Debate (pros/cons analysis)
2. **Synthesis**: Mixture of Agents (best reasoning)
3. **Validation**: Reflection (sanity check)

### Conversational AI / Chatbots
1. **Primary**: Handoff (specialist routing)
2. **Routing**: Router (intent classification)
3. **Safety**: Guardrails (content policy)

### Complex Projects
1. **Primary**: Hierarchical (manager → workers)
2. **Planning**: Plan-and-Execute (project phases)
3. **Coordination**: Agent-as-Tool (reusable skills)

---

## Pattern Combinations

Some patterns work great together:

### Router + ReAct
```python
# Route to specialist, who then uses ReAct for exploration
router.route() → specialist_react_agent.explore()
```

### Plan-and-Execute + Reflection
```python
# Plan the work, execute with quality checks
planner.plan() → executor.execute() → reflector.improve()
```

### Hierarchical + Debate
```python
# Director coordinates debaters for decision-making
director.coordinate([debater1, debater2, debater3])
```

### Parallel + Guardrails
```python
# Multiple agents work in parallel, all checked by guardrails
parallel_agents.run() → guardrails.check_all()
```

### Handoff + Agent-as-Tool
```python
# Agent A hands off to Agent B, who has other agents as tools
agentA.handoff_to(agentB_with_tool_agents)
```

---

## Anti-Patterns (What NOT to Do)

### ❌ Over-Engineering
```python
# DON'T: Use complex orchestration for simple tasks
# BAD: Hierarchical + Debate + Reflection for "What's 2+2?"
# GOOD: Just use ChatCaptain
```

### ❌ Swarm Everything
```python
# DON'T: Use swarm when you need deterministic behavior
# BAD: Swarm for financial calculations
# GOOD: Plan-and-Execute or Sequential
```

### ❌ Too Many Layers
```python
# DON'T: Nest hierarchies too deeply (>3 levels)
# BAD: Director → Manager → Supervisor → Worker
# GOOD: Director → Workers (keep it flat)
```

### ❌ Infinite Loops
```python
# DON'T: Allow unbounded iterations
# BAD: Reflection with no max_iterations
# GOOD: Always set reasonable limits (3-5 iterations)
```

### ❌ Cost Explosions
```python
# DON'T: Use expensive patterns for high-volume tasks
# BAD: Mixture of Agents for every user query
# GOOD: Use cheaper patterns, MoA only for critical decisions
```

---

## Performance Characteristics

### Latency Ranking (Fast → Slow)
1. Sequential ✅
2. Router
3. Handoff
4. Agent-as-Tool
5. Parallel ✅
6. ReAct
7. Plan-and-Execute
8. Hierarchical
9. Reflection
10. Debate
11. Swarm
12. Mixture of Agents

### Cost Ranking (Cheap → Expensive)
1. Sequential ✅
2. Router
3. Handoff
4. Agent-as-Tool
5. ReAct
6. Parallel ✅
7. Plan-and-Execute
8. Reflection
9. Hierarchical
10. Debate
11. Swarm
12. Mixture of Agents

### Quality Ranking (Good → Excellent)
1. Sequential ✅
2. Parallel ✅
3. Router
4. Handoff
5. Agent-as-Tool
6. ReAct
7. Plan-and-Execute
8. Hierarchical
9. Swarm
10. Debate
11. Reflection
12. Mixture of Agents

---

## Migration Path

For existing Fleet users, here's how to upgrade:

### Currently Using Sequential?
- ✅ Stay with Sequential for pipelines
- Consider **Plan-and-Execute** if tasks are complex
- Consider **Hierarchical** if scale is large

### Currently Using Parallel?
- ✅ Stay with Parallel for multi-perspective
- Consider **Debate** if you want argumentation
- Consider **Mixture of Agents** for highest quality
- Consider **Swarm** for true collaboration

### Want to Add Routing?
- Add **Router** for user-facing apps
- Add **Handoff** for conversational flow
- Add **Hierarchical** for task decomposition

### Want Better Quality?
- Add **Reflection** (easiest win)
- Add **Debate** (for decisions)
- Add **Mixture of Agents** (for critical outputs)

---

## Testing & Validation

### How to Test Each Pattern

**Sequential/Parallel** ✅: Simple - check final output
**ReAct**: Verify reasoning steps make sense
**Plan-and-Execute**: Validate plan quality separately
**Reflection**: Check improvement across iterations
**Hierarchical**: Test task decomposition logic
**Debate**: Ensure diverse perspectives emerge
**Router**: Test routing accuracy (confusion matrix)
**Handoff**: Verify context preservation
**Guardrails**: Test with policy violations

---

## Cost Optimization Tips

1. **Use cheaper models for routing/planning**: GPT-4o-mini for router, GPT-4o for execution
2. **Cache when possible**: Reuse plans, cache specialist responses
3. **Set iteration limits**: Reflection max 3, Debate max 2 rounds
4. **Use streaming**: Show progress, allow early termination
5. **Parallel with budget**: Limit parallel agents to 3-5
6. **Fallback strategies**: Start cheap, escalate if needed

---

## Monitoring & Observability

What to track for each pattern:

- **All Patterns**: Latency, cost, success rate
- **ReAct**: Steps taken, tool calls made
- **Plan-and-Execute**: Plan quality, execution success rate
- **Reflection**: Improvement delta per iteration
- **Hierarchical**: Task decomposition quality
- **Debate**: Consensus/dissent ratio
- **Router**: Routing accuracy, specialist utilization
- **Handoff**: Handoff chain length, context preservation

---

## Production Readiness Checklist

Before deploying a pattern to production:

- [ ] Set reasonable timeouts
- [ ] Add error handling and retries
- [ ] Implement cost limits
- [ ] Add logging/tracing
- [ ] Test edge cases
- [ ] Add guardrails if customer-facing
- [ ] Monitor performance metrics
- [ ] Have fallback strategy
- [ ] Document expected behavior
- [ ] Load test at scale

---

**Last Updated**: 2025-11-04
**Companion to**: AGENT_ORCHESTRATION_RESEARCH.md
