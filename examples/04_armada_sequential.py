"""
Example 4: Armada - Sequential Composition

This example shows how to:
- Create an Armada of multiple captains
- Execute them sequentially (chain of thought)
- Have each captain build on previous responses
"""

import os
from fleet import ChatCaptain, Armada, OpenAIProvider
import openai

def main():
    print("=== Armada Sequential Example ===\n")

    # Initialize provider
    client = openai.Client(api_key=os.getenv("OPENAI_API_KEY"))
    provider = OpenAIProvider(client)

    # Create specialized captains
    researcher = ChatCaptain(
        provider=provider,
        name="Research Captain",
        system_prompt="You are a research specialist. Gather and present factual information.",
        default_model="gpt-4o-mini"
    )

    analyst = ChatCaptain(
        provider=provider,
        name="Analysis Captain",
        system_prompt="You are an analyst. Analyze information and identify patterns and insights.",
        default_model="gpt-4o-mini"
    )

    writer = ChatCaptain(
        provider=provider,
        name="Writer Captain",
        system_prompt="You are a skilled writer. Transform analysis into clear, engaging prose.",
        default_model="gpt-4o-mini"
    )

    # Create an armada (sequential pipeline)
    armada = AgentFleet(
        captains=[researcher, analyst, writer],
        name="Content Creation Fleet",
        description="Research → Analyze → Write pipeline"
    )

    # Execute a sequential voyage
    topic = "The impact of AI on software development"

    result = armada.voyage(
        message=f"Topic: {topic}",
        model="gpt-4o-mini",
        mode="sequential",
        temperature=0.7,
        max_tokens=500
    )

    print("\n" + "="*60)
    print("FINAL OUTPUT")
    print("="*60)
    print(f"\n{result.content}\n")
    print(f"Total tokens: {result.total_tokens}")


if __name__ == "__main__":
    main()
