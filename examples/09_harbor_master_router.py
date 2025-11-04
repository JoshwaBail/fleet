"""
Example 9: Harbor Master (Router/Triage Pattern)

The Harbor Master routes incoming requests to appropriate specialist captains,
like a harbor master directing ships to the correct dock.

Use Case: Customer service, multi-domain applications, intelligent routing.
"""

import os
from fleet import HarborMaster, ChatCaptain, create_provider


def main():
    print("="*70)
    print("HARBOR MASTER (ROUTER/TRIAGE PATTERN) EXAMPLE")
    print("="*70)

    # Create provider
    provider = create_provider("openai", api_key=os.getenv("OPENAI_API_KEY"))

    # Create specialist captains
    tech_support = ChatCaptain(
        provider=provider,
        name="Tech Support Specialist",
        system_prompt="You are a technical support specialist. Help users troubleshoot technical issues.",
        description="Handles technical problems, bugs, and troubleshooting",
        default_model="gpt-4o-mini"
    )

    billing_support = ChatCaptain(
        provider=provider,
        name="Billing Specialist",
        system_prompt="You are a billing specialist. Help users with payments, invoices, and subscriptions.",
        description="Handles billing, payments, refunds, and subscription issues",
        default_model="gpt-4o-mini"
    )

    product_expert = ChatCaptain(
        provider=provider,
        name="Product Expert",
        system_prompt="You are a product expert. Help users understand features and best practices.",
        description="Handles questions about features, usage, and best practices",
        default_model="gpt-4o-mini"
    )

    # Create Harbor Master router
    harbor_master = HarborMaster(
        specialists={
            "technical": tech_support,
            "billing": billing_support,
            "product": product_expert
        },
        name="Support Router",
        description="Routes customer inquiries to the right specialist",
        routing_strategy="llm",
        router_provider=provider,
        router_model="gpt-4o-mini",
        fallback_specialist="product"
    )

    print(f"\n🏛️  Harbor Master initialized with {len(harbor_master)} specialists:\n")
    for key, specialist in harbor_master.specialists.items():
        print(f"  • {key}: {specialist.name}")

    # Test various requests
    test_requests = [
        "I can't log into my account, getting error 500",
        "I was charged twice for my subscription this month",
        "How do I export my data to CSV format?",
        "What's the difference between the Pro and Enterprise plans?"
    ]

    for i, request in enumerate(test_requests, 1):
        print(f"\n\n{'='*70}")
        print(f"REQUEST {i}")
        print("="*70)
        print(f"User: {request}")

        # Route and handle
        result = harbor_master.route(request, show_routing=True)

        print(f"\nResponse: {result.content}\n")
        print(f"Routed to: {result.metadata['specialist_name']}")
        print(f"Tokens: {result.total_tokens}")

    # Show routing statistics
    print(f"\n\n{'='*70}")
    print("ROUTING STATISTICS")
    print("="*70)
    stats = harbor_master.get_stats()
    for key, count in stats.items():
        if key != "total":
            percentage = (count / stats["total"] * 100) if stats["total"] > 0 else 0
            print(f"  {key}: {count} ({percentage:.1f}%)")
    print(f"\nTotal Requests: {stats['total']}")


if __name__ == "__main__":
    main()
