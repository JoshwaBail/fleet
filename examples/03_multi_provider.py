"""
Example 3: Using Multiple Providers

This example shows how to:
- Use different providers (OpenAI, Anthropic, OpenRouter)
- Switch between providers easily
- Compare responses from different models
"""

import os
from fleet import ChatCaptain, OpenAIProvider, AnthropicProvider, OpenRouterProvider
import openai
import anthropic

def main():
    print("=== Multi-Provider Example ===\n")

    # Initialize different providers
    openai_provider = OpenAIProvider(
        openai.Client(api_key=os.getenv("OPENAI_API_KEY"))
    )

    anthropic_provider = AnthropicProvider(
        anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    )

    openrouter_provider = OpenRouterProvider(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        site_name="Fleet Example"
    )

    # Create captains with different providers
    openai_captain = ChatCaptain(
        provider=openai_provider,
        name="OpenAI Navigator",
        system_prompt="You are a concise AI assistant.",
        default_model="gpt-4o-mini"
    )

    anthropic_captain = ChatCaptain(
        provider=anthropic_provider,
        name="Claude Navigator",
        system_prompt="You are a concise AI assistant.",
        default_model="claude-3-5-haiku-20241022"
    )

    openrouter_captain = ChatCaptain(
        provider=openrouter_provider,
        name="OpenRouter Navigator",
        system_prompt="You are a concise AI assistant.",
        default_model="meta-llama/llama-3.1-8b-instruct"
    )

    # Ask the same question to all captains
    question = "Explain quantum computing in one sentence."

    print(f"Question: {question}\n")

    # OpenAI response
    print("--- OpenAI (GPT-4o-mini) ---")
    try:
        response = openai_captain.chat(question)
        print(f"{response.content}")
        print(f"Tokens: {response.total_tokens}\n")
    except Exception as e:
        print(f"Error: {e}\n")

    # Anthropic response
    print("--- Anthropic (Claude 3.5 Haiku) ---")
    try:
        response = anthropic_captain.chat(question)
        print(f"{response.content}")
        print(f"Tokens: {response.total_tokens}\n")
    except Exception as e:
        print(f"Error: {e}\n")

    # OpenRouter response
    print("--- OpenRouter (Llama 3.1) ---")
    try:
        response = openrouter_captain.chat(question)
        print(f"{response.content}")
        print(f"Tokens: {response.total_tokens}\n")
    except Exception as e:
        print(f"Error: {e}\n")


if __name__ == "__main__":
    main()
