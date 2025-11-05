"""
Example 11: Council - Multi-Agent Debate Pattern

This example demonstrates the Council debate pattern where multiple agents
with different perspectives debate a topic and reach a resolution through
synthesis, voting, or consensus.

Use Case: Product strategy decision-making where multiple perspectives
(technical, business, customer) need to be considered and synthesized.

Pattern: Debate (Multi-Agent Deliberation)
- Multiple agents with distinct perspectives
- Structured debate rounds (opening statements + rebuttals)
- Resolution strategies: synthesis, vote, or consensus
- Useful for important decisions requiring multiple viewpoints
"""

import os
from fleet import Council, ChatCaptain
from fleet.providers import create_provider


def main():
    print("=" * 70)
    print("COUNCIL DEBATE PATTERN")
    print("Multi-Agent Deliberation for Product Strategy")
    print("=" * 70)
    print()

    # Setup provider
    provider = create_provider("openai", api_key=os.getenv("OPENAI_API_KEY"))

    # Create council members with different perspectives
    print("⚓ Assembling the Council...")
    print()

    # Technical perspective
    tech_captain = ChatCaptain(
        provider=provider,
        default_model="gpt-4o-mini",
        system_prompt=(
            "You are a Technical Lead focused on technical feasibility, "
            "scalability, maintainability, and engineering best practices. "
            "Be pragmatic about technical constraints and implementation complexity. "
            "Your priority is building robust, maintainable systems."
        ),
        name="TechLead"
    )

    # Business perspective
    business_captain = ChatCaptain(
        provider=provider,
        default_model="gpt-4o-mini",
        system_prompt=(
            "You are a Business Strategist focused on market opportunity, "
            "revenue potential, competitive advantage, and business viability. "
            "Consider ROI, time-to-market, and strategic positioning. "
            "Your priority is maximizing business value."
        ),
        name="BusinessStrategist"
    )

    # Customer perspective
    customer_captain = ChatCaptain(
        provider=provider,
        default_model="gpt-4o-mini",
        system_prompt=(
            "You are a Customer Experience Advocate focused on user needs, "
            "usability, accessibility, and customer satisfaction. "
            "Consider user pain points, adoption barriers, and delightful experiences. "
            "Your priority is creating value for end users."
        ),
        name="CustomerAdvocate"
    )

    # Create the council
    council = Council(
        debaters=[tech_captain, business_captain, customer_captain],
        provider=provider,
        model="gpt-4o-mini",
        resolution_strategy="synthesis",  # or "vote" or "consensus"
        max_rounds=2,
        name="ProductStrategyCouncil"
    )

    # Debate topic
    topic = """
    Should we build a new AI-powered code review feature that:
    - Uses LLM to analyze pull requests for bugs, style issues, and best practices
    - Provides automated suggestions and explanations
    - Learns from team's code review patterns
    - Estimated development time: 3-4 months
    - Estimated cost: $200K in engineering resources

    Debate: Should we prioritize this feature in our next quarter?
    """

    print("📋 Topic for Debate:")
    print(topic.strip())
    print()
    print("=" * 70)
    print()

    # Convene the council
    print("🏛️  Convening the Council...")
    print("   Resolution Strategy: synthesis")
    print("   Max Rounds: 2 (opening + 1 rebuttal)")
    print()

    result = council.convene(
        topic=topic,
        temperature=0.7,
        max_tokens=500
    )

    # Display results
    print()
    print("=" * 70)
    print("COUNCIL DECISION")
    print("=" * 70)
    print()
    print(result.content)
    print()

    # Show metadata
    if "debate_rounds" in result.metadata:
        print("=" * 70)
        print("DEBATE STATISTICS")
        print("=" * 70)
        print(f"Total Rounds: {len(result.metadata['debate_rounds'])}")
        print(f"Total Tokens: {result.total_tokens:,}")
        print(f"Resolution Strategy: {result.metadata.get('resolution_strategy', 'N/A')}")
        print()

    # Example of different resolution strategies
    print("=" * 70)
    print("TRYING DIFFERENT RESOLUTION STRATEGIES")
    print("=" * 70)
    print()

    # Vote-based resolution
    print("🗳️  Vote Resolution (majority wins):")
    council_vote = Council(
        debaters=[tech_captain, business_captain, customer_captain],
        provider=provider,
        model="gpt-4o-mini",
        resolution_strategy="vote",
        max_rounds=1,  # Just opening statements for quick demo
        name="VotingCouncil"
    )

    vote_result = council_vote.convene(
        topic="Quick vote: Should we prioritize AI code review feature? (Yes/No)",
        temperature=0.7,
        max_tokens=300
    )
    print(f"Result: {vote_result.content[:200]}...")
    print()

    print("=" * 70)
    print("WHEN TO USE COUNCIL PATTERN")
    print("=" * 70)
    print("""
    ✅ Use Council When:
    - Important decisions with multiple valid perspectives
    - Trade-offs need thorough exploration
    - Diverse expertise required
    - Consensus or synthesis desired
    - Quality matters more than speed

    ❌ Avoid Council When:
    - Simple decisions with clear answers
    - Speed is critical
    - Single perspective sufficient
    - Cost is primary concern

    💡 Tips:
    - Give each debater a clear, distinct perspective
    - Use synthesis for balanced decisions
    - Use vote for binary decisions
    - Use consensus when agreement is crucial
    - Limit rounds (2-3) to control cost
    - Start with opening statements to gauge positions
    """)


if __name__ == "__main__":
    main()
