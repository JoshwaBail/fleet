"""
Example 8: Quartermaster (Reflection Pattern)

The Quartermaster uses the Reflection pattern: generating output,
critiquing it, and iteratively improving for higher quality.

Use Case: Content creation where quality is more important than speed.
"""

import os
from fleet import ReflectiveCaptain, create_provider


def main():
    print("="*70)
    print("QUARTERMASTER (REFLECTION PATTERN) EXAMPLE")
    print("="*70)

    # Create provider
    provider = create_provider("openai", api_key=os.getenv("OPENAI_API_KEY"))

    # Create Quartermaster
    quartermaster = ReflectiveCaptain(
        provider=provider,
        name="Content Refiner",
        system_prompt="You are a skilled technical writer focused on clarity and accuracy.",
        critique_prompt="""Review the previous output critically for:
1. Technical accuracy
2. Clarity and coherence
3. Completeness of explanation
4. Appropriate depth for the audience
5. Grammar and style

Be specific about what needs improvement.""",
        default_model="gpt-4o-mini",
        max_iterations=3,
        auto_improve=True,
        verbose=True
    )

    # Writing task
    task = """Write a 2-paragraph explanation of how Kubernetes handles container orchestration.
Target audience: Software developers new to Kubernetes."""

    print(f"\nTask: {task}\n")

    # Reflect and improve
    result = quartermaster.reflect(task)

    print("\n" + "="*70)
    print("FINAL VERSION (After Reflection)")
    print("="*70)
    print(result.content)

    # Show improvement over iterations
    print(f"\n\n{'='*70}")
    print("IMPROVEMENT HISTORY")
    print("="*70)

    iterations = result.metadata.get("all_iterations", [])
    for i, iteration in enumerate(iterations, 1):
        print(f"\n--- Iteration {i} ---")
        if i == 1:
            print("(Initial draft)")
        print(f"\nContent:\n{iteration['content'][:150]}...")

        if iteration.get('critique'):
            print(f"\nCritique:\n{iteration['critique'][:150]}...")

    print(f"\n\nTotal Tokens: {result.total_tokens}")


if __name__ == "__main__":
    main()
