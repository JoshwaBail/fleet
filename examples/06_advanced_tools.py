"""
Example 6: Advanced Tool Usage

This example shows:
- Building complex arsenals
- Using tools with different data types
- Chaining tool calls
- Error handling in tools
"""

import os
from fleet import ToolCaptain, OpenAIProvider, Instrument, InstrumentParameter, Toolbox
import openai
import json
from datetime import datetime

def main():
    print("=== Advanced Tools Example ===\n")

    # Initialize provider
    client = openai.Client(api_key=os.getenv("OPENAI_API_KEY"))
    provider = OpenAIProvider(client)

    # Create instruments with explicit parameters
    search_ports = Tool.from_function(
        name="search_ports",
        description="Search for ports by name or region",
        function=lambda query, region="all": {
            "query": query,
            "results": [
                {"name": "Port of Boston", "region": "Northeast", "capacity": "large"},
                {"name": "Port of Miami", "region": "Southeast", "capacity": "large"},
                {"name": "Port of Seattle", "region": "Northwest", "capacity": "medium"}
            ]
        },
        parameters=[
            InstrumentParameter("query", "string", "Search query", required=True),
            InstrumentParameter("region", "string", "Region filter", required=False, default="all")
        ]
    )

    get_port_details = Tool.from_function(
        name="get_port_details",
        description="Get detailed information about a specific port",
        function=lambda port_name: {
            "name": port_name,
            "coordinates": {"lat": 42.3601, "lon": -71.0589},
            "depth": "40 feet",
            "facilities": ["container", "bulk", "passenger"],
            "operational": True,
            "last_updated": datetime.now().isoformat()
        }
    )

    calculate_route = Tool.from_function(
        name="calculate_route",
        description="Calculate optimal route between ports",
        function=lambda origin, destination, avoid_storms=True: {
            "origin": origin,
            "destination": destination,
            "distance_nm": 245,
            "estimated_time_hours": 12,
            "fuel_required_gallons": 150,
            "waypoints": 3,
            "storm_avoidance": avoid_storms
        },
        parameters=[
            InstrumentParameter("origin", "string", "Starting port", required=True),
            InstrumentParameter("destination", "string", "Destination port", required=True),
            InstrumentParameter("avoid_storms", "boolean", "Avoid storm systems", required=False, default=True)
        ]
    )

    # Build arsenal
    navigation_toolbox = Toolbox("Advanced Navigation")
    navigation_toolbox.add_tool(search_ports)
    navigation_toolbox.add_tool(get_port_details)
    navigation_toolbox.add_tool(calculate_route)

    # Create captain
    captain = ToolCaptain(
        provider=provider,
        name="Master Navigator",
        system_prompt="""You are an expert ship navigator. Use your tools to:
1. Search for appropriate ports
2. Get detailed port information
3. Calculate optimal routes
Provide comprehensive answers with specific details from your tools.""",
        arsenal=navigation_arsenal,
        default_model="gpt-4o-mini",
        max_tool_rounds=10  # Allow multiple rounds of tool calling
    )

    # Complex query requiring multiple tool calls
    query = """I need to plan a voyage from Boston to a large port in the Southeast region.
Please find suitable destination ports, get details about the best option,
and calculate the optimal route."""

    print(f"Query: {query}\n")
    print("="*60)

    response = captain.chat(query)

    print(f"\nCaptain's Response:\n{response.content}\n")
    print(f"Tokens used: {response.total_tokens}")
    print(f"Messages in history: {len(captain.get_messages())}")


if __name__ == "__main__":
    main()
