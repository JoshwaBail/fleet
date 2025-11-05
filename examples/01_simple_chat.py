"""
Example 1: Simple Chat with ChatCaptain

This example shows the most basic usage of Fleet:
- Create a provider
- Create a ChatCaptain
- Have a conversation
"""

import os
from fleet import ChatCaptain, OpenAIProvider
import openai

def main():
    # Initialize provider
    client = openai.Client(api_key=os.getenv("OPENAI_API_KEY"))
    provider = OpenAIProvider(client)

    # Create a simple chat captain
    captain = ChatCaptain(
        provider=provider,
        name="Navigator",
        system_prompt="You are a helpful AI assistant with a nautical theme. Use ship and sailing metaphors when appropriate.",
        default_model="gpt-4o-mini",
        default_temperature=0.7
    )

    # Have a conversation
    print("=== Simple Chat Example ===\n")

    response1 = captain.chat("What's the weather like for sailing today?")
    print(f"Captain: {response1.content}\n")
    print(f"Tokens used: {response1.total_tokens}\n")

    # Continue the conversation (maintains context)
    response2 = captain.chat("What should I pack for the voyage?")
    print(f"Captain: {response2.content}\n")
    print(f"Tokens used: {response2.total_tokens}\n")

    # Check conversation history
    print(f"Total messages in history: {len(captain.get_messages())}")


if __name__ == "__main__":
    main()
