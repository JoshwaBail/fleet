"""
Example 0: Cookie-Cutter Provider Usage

This example demonstrates the TRUE power of Fleet's dependency injection:
All providers work EXACTLY the same way. Just change the provider name!

This is the recommended way to use Fleet.
"""

import os
from fleet import ChatCaptain, create_provider

def main():
    print("=== Cookie-Cutter Provider Example ===\n")
    print("Fleet's dependency injection makes switching providers trivial!\n")

    # ============================================================================
    # METHOD 1: Factory Function (Recommended - Most Flexible)
    # ============================================================================
    print("--- Method 1: Using create_provider() factory ---\n")

    # Just change the first argument to switch providers!
    # Everything else stays EXACTLY the same.

    # OpenAI
    print("Creating OpenAI provider...")
    openai_provider = create_provider("openai", api_key=os.getenv("OPENAI_API_KEY"))

    # Anthropic (Claude)
    print("Creating Anthropic provider...")
    anthropic_provider = create_provider("anthropic", api_key=os.getenv("ANTHROPIC_API_KEY"))

    # OpenRouter (Access to 50+ models)
    print("Creating OpenRouter provider...")
    openrouter_provider = create_provider("openrouter", api_key=os.getenv("OPENROUTER_API_KEY"))

    print("\n" + "="*60 + "\n")

    # ============================================================================
    # METHOD 2: Direct Instantiation (Also Cookie-Cutter)
    # ============================================================================
    print("--- Method 2: Direct instantiation ---\n")

    from fleet import OpenAIProvider, AnthropicProvider, OpenRouterProvider

    # All three have IDENTICAL signatures!
    provider_a = OpenAIProvider(api_key=os.getenv("OPENAI_API_KEY"))
    provider_b = AnthropicProvider(api_key=os.getenv("ANTHROPIC_API_KEY"))
    provider_c = OpenRouterProvider(api_key=os.getenv("OPENROUTER_API_KEY"))

    print("All providers created with identical syntax!\n")
    print("="*60 + "\n")

    # ============================================================================
    # Usage is IDENTICAL for all providers
    # ============================================================================
    print("--- Testing with different providers ---\n")

    question = "In one sentence, what is quantum computing?"

    # Test with OpenAI
    try:
        print("🤖 OpenAI (GPT-4o-mini):")
        captain = ChatCaptain(
            provider=openai_provider,
            name="OpenAI Bot",
            default_model="gpt-4o-mini"
        )
        response = captain.chat(question)
        print(f"   {response.content}\n")
    except Exception as e:
        print(f"   Error: {e}\n")

    # Test with Anthropic (same code, different provider!)
    try:
        print("🤖 Anthropic (Claude 3.5 Haiku):")
        captain = ChatCaptain(
            provider=anthropic_provider,
            name="Claude Bot",
            default_model="claude-3-5-haiku-20241022"
        )
        response = captain.chat(question)
        print(f"   {response.content}\n")
    except Exception as e:
        print(f"   Error: {e}\n")

    # Test with OpenRouter (same code, different provider!)
    try:
        print("🤖 OpenRouter (Llama 3.1 8B):")
        captain = ChatCaptain(
            provider=openrouter_provider,
            name="Llama Bot",
            default_model="meta-llama/llama-3.1-8b-instruct"
        )
        response = captain.chat(question)
        print(f"   {response.content}\n")
    except Exception as e:
        print(f"   Error: {e}\n")

    print("="*60)
    print("\n✨ THAT'S DEPENDENCY INJECTION!")
    print("   Same code, different providers. Switch anytime.\n")

    # ============================================================================
    # Configuration-Driven Provider Selection
    # ============================================================================
    print("--- Bonus: Config-driven provider selection ---\n")

    # This pattern is perfect for environment-based configuration
    config = {
        "provider": os.getenv("LLM_PROVIDER", "openai"),  # From env or config file
        "api_key": os.getenv("LLM_API_KEY"),
        "model": os.getenv("LLM_MODEL", "gpt-4o-mini")
    }

    # Create provider from config
    provider = create_provider(config["provider"], api_key=config["api_key"])
    captain = ChatCaptain(provider=provider, default_model=config["model"])

    print(f"Using provider: {config['provider']}")
    print(f"Using model: {config['model']}")
    print("\nThis makes Fleet perfect for:")
    print("  • Multi-tenant applications")
    print("  • A/B testing different models")
    print("  • Cost optimization by provider")
    print("  • Fallback strategies")
    print("  • Environment-based configuration\n")


if __name__ == "__main__":
    main()
