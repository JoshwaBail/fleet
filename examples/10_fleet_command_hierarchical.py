"""
Example 10: Fleet Command (Hierarchical Pattern)

Fleet Command implements hierarchical orchestration where a director
coordinates multiple specialized workers.

Use Case: Complex projects requiring task decomposition and specialized expertise.
"""

import os
from fleet import FleetCommand, ChatCaptain, create_provider


def main():
    print("="*70)
    print("FLEET COMMAND (HIERARCHICAL PATTERN) EXAMPLE")
    print("="*70)

    # Create provider
    provider = create_provider("openai", api_key=os.getenv("OPENAI_API_KEY"))

    # Create Director
    director = ChatCaptain(
        provider=provider,
        name="Project Director",
        system_prompt="""You are a project director. You excel at:
1. Breaking down complex projects into tasks
2. Assigning tasks to team members based on their expertise
3. Synthesizing team contributions into cohesive deliverables

Be clear and systematic in your planning.""",
        description="Coordinates the team and manages project execution",
        default_model="gpt-4o-mini"
    )

    # Create specialized workers
    researcher = ChatCaptain(
        provider=provider,
        name="Research Analyst",
        system_prompt="You are a research analyst. You gather information, analyze data, and provide insights.",
        description="Conducts research and data analysis",
        default_model="gpt-4o-mini"
    )

    architect = ChatCaptain(
        provider=provider,
        name="Solutions Architect",
        system_prompt="You are a solutions architect. You design technical architectures and system designs.",
        description="Designs technical solutions and architectures",
        default_model="gpt-4o-mini"
    )

    writer = ChatCaptain(
        provider=provider,
        name="Technical Writer",
        system_prompt="You are a technical writer. You create clear, well-structured documentation.",
        description="Writes documentation and technical content",
        default_model="gpt-4o-mini"
    )

    # Create Fleet Command
    fleet_command = FleetCommand(
        director=director,
        workers=[researcher, architect, writer],
        name="Project Team",
        description="Collaborative project team",
        task_decomposition_strategy="llm",
        synthesize_results=True
    )

    print(f"\nFleet Command assembled:")
    print(f"  Director: {director.name}")
    print(f"  Workers: {', '.join(w.name for w in fleet_command.workers)}\n")

    # Complex mission
    mission = """Design a recommendation system for an e-commerce platform.
The system should provide personalized product recommendations based on user behavior.
Include research on approaches, technical architecture, and implementation guide."""

    print(f"Mission: {mission}\n")

    # Execute the mission
    result = fleet_command.execute_mission(
        mission=mission,
        model="gpt-4o-mini",
        temperature=0.7,
        max_tokens=500
    )

    print(f"\n{'='*70}")
    print("FINAL DELIVERABLE")
    print("="*70)
    print(result.content)

    print(f"\n\n{'='*70}")
    print("PROJECT METADATA")
    print("="*70)
    print(f"Workers Involved: {result.metadata.get('worker_count', 0)}")
    print(f"Successful Tasks: {result.metadata.get('successful_workers', 0)}")
    print(f"Total Tokens: {result.total_tokens}")


if __name__ == "__main__":
    main()
