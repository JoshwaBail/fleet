"""
Example 13: Admiral - Plan-and-Execute Pattern

This example demonstrates the Admiral pattern which separates planning from execution.
The Admiral first creates a detailed plan, then executes each step systematically.
Can use different models for planning vs execution (e.g., expensive planner, cheap executor).

Use Case: Complex research and analysis task where upfront planning improves
execution quality and allows for cost optimization by using different model tiers.

Pattern: Plan-and-Execute
- Separate planning phase
- Step-by-step execution
- Different models for planner vs executor
- Execution tracking and status
- Replanning support on failures
- Useful for complex, multi-step tasks
"""

import os
from fleet import Admiral
from fleet.providers import create_provider
from fleet.instruments import Instrument, Arsenal


def main():
    print("=" * 70)
    print("ADMIRAL PLAN-AND-EXECUTE PATTERN")
    print("Complex Research Task with Separate Planning")
    print("=" * 70)
    print()

    # Setup provider
    provider = create_provider("openai", api_key=os.getenv("OPENAI_API_KEY"))

    # Create research tools
    def search_academic_papers(query: str, year: int = 2024) -> str:
        """Search for academic papers on a topic"""
        # Simulated academic search
        papers = {
            "agent orchestration": [
                "AutoGen: Enabling Next-Gen LLM Applications (2024)",
                "LangGraph: Multi-Agent Workflows (2024)",
                "ReAct: Synergizing Reasoning and Acting (2023)"
            ],
            "multi-agent systems": [
                "Cooperation and Competition in Multi-Agent Systems (2024)",
                "Hierarchical Multi-Agent Planning (2024)",
                "Swarm Intelligence Patterns (2023)"
            ],
            "llm applications": [
                "Large Language Models in Production (2024)",
                "Cost-Effective LLM Deployment (2024)",
                "Prompt Engineering at Scale (2023)"
            ]
        }
        for key in papers:
            if key in query.lower():
                return f"Found {len(papers[key])} papers: " + ", ".join(papers[key])
        return "No papers found for this query."

    def analyze_market_trends(industry: str, timeframe: str = "2024") -> str:
        """Analyze market trends for an industry"""
        # Simulated market analysis
        trends = {
            "ai": "Growing 45% YoY, focus on enterprise adoption and cost efficiency",
            "saas": "Consolidation phase, focus on profitability over growth",
            "developer tools": "AI-powered tools dominating, 30% market growth"
        }
        return trends.get(industry.lower(), "Limited data available")

    def gather_user_feedback(product_category: str) -> str:
        """Gather user feedback and pain points"""
        # Simulated user feedback
        feedback = {
            "agent frameworks": "Users want: 1) Simpler APIs, 2) Better debugging, 3) Lower costs, 4) Multi-provider support",
            "developer tools": "Users want: 1) Faster iteration, 2) Better documentation, 3) Real-world examples",
            "ai tools": "Users want: 1) Transparency, 2) Control over costs, 3) Reliability"
        }
        return feedback.get(product_category.lower(), "No feedback data available")

    def calculate_opportunity_score(market_size: str, competition: str, user_demand: str) -> str:
        """Calculate opportunity score based on factors"""
        # Simulated scoring
        return f"Opportunity Score: 7.5/10 (High market demand, moderate competition, strong user need)"

    # Create arsenal
    arsenal = Arsenal()
    arsenal.add_instrument(Instrument.from_function(search_academic_papers))
    arsenal.add_instrument(Instrument.from_function(analyze_market_trends))
    arsenal.add_instrument(Instrument.from_function(gather_user_feedback))
    arsenal.add_instrument(Instrument.from_function(calculate_opportunity_score))

    print("⚓ Creating Admiral with research instruments...")
    print()

    # Create Admiral
    # In production, you might use gpt-4 for planning and gpt-4o-mini for execution
    admiral = Admiral(
        provider=provider,
        instruments=arsenal.get_instruments(),
        system_prompt=(
            "You are a strategic research analyst. Create detailed, actionable plans "
            "and execute them systematically using available research tools. "
            "Each step should build on previous findings."
        ),
        name="ResearchAdmiral"
    )

    # Mission: Comprehensive market research
    mission = """
    Conduct a comprehensive analysis of the AI agent framework market to determine
    if there's an opportunity for a new lightweight, multi-provider agent framework.

    Your analysis should cover:
    1. Current academic research and state-of-the-art
    2. Market trends and growth
    3. User needs and pain points
    4. Opportunity assessment

    Provide a final recommendation with supporting evidence.
    """

    print("📋 Mission Brief:")
    print(mission.strip())
    print()
    print("=" * 70)
    print()

    # Execute the mission
    # Using different models for planning vs execution
    print("🎯 Executing Mission with Plan-and-Execute Pattern...")
    print()

    result = admiral.command(
        mission=mission,
        planner_model="gpt-4o-mini",  # Planner creates the strategy
        executor_model="gpt-4o-mini",  # Executor carries out each step
        temperature=0.3,  # Lower temperature for more focused execution
        max_tokens=800
    )

    print()
    print("=" * 70)
    print("MISSION RESULTS")
    print("=" * 70)
    print()
    print(result.content)
    print()

    # Show execution details
    if "execution_steps" in result.metadata:
        steps = result.metadata["execution_steps"]
        print("=" * 70)
        print("EXECUTION DETAILS")
        print("=" * 70)
        print()
        print(f"Total Steps: {len(steps)}")
        print()

        for i, step in enumerate(steps, 1):
            print(f"Step {i}: {step.get('description', 'N/A')}")
            print(f"  Status: {step.get('status', 'unknown')}")
            if step.get('tool_used'):
                print(f"  Tool Used: {step['tool_used']}")
            print()

    # Show the original plan
    if "plan" in result.metadata:
        print("=" * 70)
        print("ORIGINAL PLAN")
        print("=" * 70)
        print()
        print(result.metadata["plan"])
        print()

    # Token usage breakdown
    print("=" * 70)
    print("RESOURCE USAGE")
    print("=" * 70)
    print(f"Total Tokens: {result.total_tokens:,}")
    if "planning_tokens" in result.metadata:
        print(f"Planning Phase: {result.metadata['planning_tokens']:,} tokens")
    if "execution_tokens" in result.metadata:
        print(f"Execution Phase: {result.metadata['execution_tokens']:,} tokens")
    print()

    # Demonstrate cost optimization
    print("=" * 70)
    print("COST OPTIMIZATION EXAMPLE")
    print("=" * 70)
    print("""
    The Admiral pattern allows for cost optimization by using different
    model tiers for different phases:

    Example Configuration:
    ┌─────────────────────────────────────────────────────────────┐
    │ Planner: gpt-4 (expensive but strategic)                    │
    │ - Creates comprehensive, high-quality plan                   │
    │ - Only runs once                                             │
    │ - Cost: ~$0.02 for planning                                  │
    └─────────────────────────────────────────────────────────────┘
                              ↓
    ┌─────────────────────────────────────────────────────────────┐
    │ Executor: gpt-4o-mini (cheap but capable)                   │
    │ - Follows the plan step-by-step                              │
    │ - Runs for each step (5-10 times)                            │
    │ - Cost: ~$0.01 total for all executions                      │
    └─────────────────────────────────────────────────────────────┘

    Total Cost: ~$0.03 vs ~$0.10 if using gpt-4 for everything
    Savings: 70% while maintaining strategic quality
    """)

    print("=" * 70)
    print("WHEN TO USE ADMIRAL PATTERN")
    print("=" * 70)
    print("""
    ✅ Use Admiral When:
    - Complex multi-step tasks benefit from upfront planning
    - Want to review/approve plan before execution
    - Can use different model tiers for optimization
    - Execution might fail and need replanning
    - Systematic approach improves outcomes

    ❌ Avoid Admiral When:
    - Simple single-step tasks
    - Highly dynamic environments (plan becomes stale)
    - Planning overhead not justified
    - Real-time responses needed

    💡 Tips:
    - Use expensive model for planning, cheap for execution
    - Keep plans focused (3-7 steps ideal)
    - Include contingency steps in plan
    - Monitor execution status
    - Enable replanning for long-running tasks

    🆚 Admiral vs Navigator:
    Admiral: Plan upfront, execute systematically
      - Better when path is knowable
      - More cost-efficient with model mixing
      - Less adaptive but more predictable

    Navigator: Reason dynamically, adapt as you go
      - Better for exploratory tasks
      - More adaptive to unexpected findings
      - Higher cost but more flexible
    """)

    # Show comparison example
    print("=" * 70)
    print("PATTERN COMPARISON")
    print("=" * 70)
    print("""
    Same Task, Different Patterns:

    Admiral (Plan-and-Execute):
    1. [Planning Phase] Create 5-step research plan
    2. [Execution] Execute step 1 → 2 → 3 → 4 → 5
    3. [Synthesis] Combine results
    ⏱️  Time: Moderate  💰 Cost: Low-Medium  🎯 Quality: High

    Navigator (ReAct):
    1. Think: "I need to research X"
    2. Act: Research X
    3. Observe: "Found Y, now I need Z"
    4. Think: "Based on Y, I should..."
    5. Act: Research Z
    6. Iterate until complete
    ⏱️  Time: Moderate  💰 Cost: Medium  🎯 Quality: High

    Simple ToolCaptain:
    1. Use all tools at once
    2. Synthesize results
    ⏱️  Time: Fast  💰 Cost: Low  🎯 Quality: Medium

    Choice depends on:
    - Task complexity
    - Need for strategic planning
    - Cost constraints
    - Adaptability requirements
    """)


if __name__ == "__main__":
    main()
