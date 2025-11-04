"""
Example 2: Using Tools with ToolCaptain

This example shows how to:
- Create instruments (tools)
- Build an arsenal (toolbox)
- Use ToolCaptain with function calling
"""

import os
from fleet import ToolCaptain, OpenAIProvider, tool, Toolbox
import openai

# Define some instruments using the decorator
@tool(
    name="get_weather",
    description="Get current weather for a location"
)
def get_weather(location: str, units: str = "celsius") -> dict:
    """Get weather for a location (simulated)"""
    # In real app, this would call a weather API
    return {
        "location": location,
        "temperature": 22 if units == "celsius" else 72,
        "units": units,
        "condition": "sunny",
        "wind_speed": "10 knots"
    }

@tool(
    name="calculate_distance",
    description="Calculate nautical distance between two ports"
)
def calculate_distance(port_a: str, port_b: str) -> dict:
    """Calculate distance between ports (simulated)"""
    # In real app, this would calculate actual distance
    return {
        "from": port_a,
        "to": port_b,
        "distance_nm": 245,  # nautical miles
        "estimated_time": "12 hours"
    }

@tool(
    name="check_supplies",
    description="Check available supplies and inventory"
)
def check_supplies(item_type: str) -> dict:
    """Check supply inventory (simulated)"""
    inventory = {
        "food": {"quantity": 50, "unit": "days", "status": "sufficient"},
        "water": {"quantity": 1000, "unit": "gallons", "status": "sufficient"},
        "fuel": {"quantity": 80, "unit": "percent", "status": "good"}
    }
    return inventory.get(item_type, {"status": "unknown"})


def main():
    print("=== Tools and Instruments Example ===\n")

    # Initialize provider
    client = openai.Client(api_key=os.getenv("OPENAI_API_KEY"))
    provider = OpenAIProvider(client)

    # Create an arsenal with our instruments
    navigation_toolbox = Toolbox(
        name="Navigation Arsenal",
        description="Tools for navigation and voyage planning"
    )
    navigation_toolbox.add_tool(get_weather)
    navigation_toolbox.add_tool(calculate_distance)
    navigation_toolbox.add_tool(check_supplies)

    print(f"Arsenal created with {len(navigation_arsenal)} instruments")
    print(f"Instruments: {', '.join(navigation_toolbox.list_instruments())}\n")

    # Create a ToolCaptain with the arsenal
    captain = ToolCaptain(
        provider=provider,
        name="Navigation Officer",
        system_prompt="You are a ship's navigation officer. Use your tools to help plan voyages and check conditions.",
        arsenal=navigation_arsenal,
        default_model="gpt-4o-mini"
    )

    # Ask a question that requires tool usage
    response = captain.chat(
        "I need to sail from Boston to New York. What's the weather like, "
        "how far is it, and do we have enough fuel?"
    )

    print(f"\nCaptain's Response:\n{response.content}\n")
    print(f"Tokens used: {response.total_tokens}")

    # Check if tools were called
    if response.tool_calls:
        print(f"\nTools called: {len(response.tool_calls)}")
        for tc in response.tool_calls:
            print(f"  - {tc['function']['name']}")


if __name__ == "__main__":
    main()
