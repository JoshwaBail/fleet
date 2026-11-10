"""One agent, one tool.

pip install -e ".[anthropic]"
export ANTHROPIC_API_KEY=...
python examples/quickstart.py
"""

from anthropic import Anthropic

from fleet import Agent


def get_weather(city: str) -> str:
    """Look up the current weather in a city.

    Args:
        city: The city's name, in English.
    """
    return {"Lisbon": "18°C and sunny", "Bergen": "9°C and raining"}.get(city, "No data for that city.")


agent = Agent(
    client=Anthropic(),
    model="claude-opus-5-5",
    instructions="You are a concise assistant.",
    tools=[get_weather],
    # Passed straight to the Messages API: if the model declines a request, let the API retry it on another model.
    params={"betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"},
)

if __name__ == "__main__":
    result = agent.run_sync("I'm flying from Lisbon to Bergen today. What should I pack?")
    print(result.output)
    print(
        f"\n({result.usage.requests} requests, {result.usage.input_tokens} tokens in, {result.usage.output_tokens} out)"
    )
