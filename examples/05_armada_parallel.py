"""
Example 5: Armada - Parallel Composition

This example shows how to:
- Execute multiple captains in parallel
- Get diverse perspectives on the same question
- Synthesize responses into a coherent answer
"""

import os
from fleet import ChatCaptain, Armada, OpenAIProvider
import openai

def main():
    print("=== Armada Parallel Example ===\n")

    # Initialize provider
    client = openai.Client(api_key=os.getenv("OPENAI_API_KEY"))
    provider = OpenAIProvider(client)

    # Create captains with different perspectives
    technical_captain = ChatCaptain(
        provider=provider,
        name="Technical Officer",
        system_prompt="You are a technical expert. Focus on technical details, architecture, and implementation.",
        default_model="gpt-4o-mini"
    )

    business_captain = ChatCaptain(
        provider=provider,
        name="Business Officer",
        system_prompt="You are a business strategist. Focus on business value, ROI, and market impact.",
        default_model="gpt-4o-mini"
    )

    ux_captain = ChatCaptain(
        provider=provider,
        name="UX Officer",
        system_prompt="You are a UX specialist. Focus on user experience, usability, and user needs.",
        default_model="gpt-4o-mini"
    )

    security_captain = ChatCaptain(
        provider=provider,
        name="Security Officer",
        system_prompt="You are a security expert. Focus on security implications, risks, and best practices.",
        default_model="gpt-4o-mini"
    )

    # Create an armada with parallel execution and synthesis
    armada = AgentFleet(
        captains=[technical_captain, business_captain, ux_captain, security_captain],
        name="Product Review Fleet",
        description="Multi-perspective product analysis",
        synthesize=True  # Combine all perspectives into one response
    )

    # Execute a parallel voyage
    question = "Should we build a mobile app with biometric authentication for our banking platform?"

    result = armada.voyage(
        message=question,
        model="gpt-4o-mini",
        mode="parallel",
        temperature=0.7,
        max_tokens=300
    )

    print("\n" + "="*60)
    print("SYNTHESIZED DECISION")
    print("="*60)
    print(f"\n{result.content}\n")
    print(f"Total tokens: {result.total_tokens}")


if __name__ == "__main__":
    main()
