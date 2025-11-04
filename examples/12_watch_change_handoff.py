"""
Example 12: WatchChange - Context-Preserving Handoff Pattern

This example demonstrates the WatchChange handoff pattern where agents can
transfer requests to specialized agents while preserving full conversation context.
Similar to OpenAI's Swarm pattern.

Use Case: Customer support escalation where a general support agent can hand off
complex technical issues to specialists, or billing questions to billing team,
while maintaining full context of the conversation.

Pattern: Handoff (Agent Transfer)
- Context preservation across agents
- Handoff tools registered as instruments
- Handoff chain tracking
- Max handoff limits to prevent loops
- Useful for progressive specialization and escalation workflows
"""

import os
from fleet import HandoffCaptain
from fleet.providers import create_provider
from fleet.tools import Instrument, Toolbox


def main():
    print("=" * 70)
    print("WATCH CHANGE HANDOFF PATTERN")
    print("Customer Support with Specialist Escalation")
    print("=" * 70)
    print()

    # Setup provider
    provider = create_provider("openai", api_key=os.getenv("OPENAI_API_KEY"))

    print("⚓ Setting up Support Team...")
    print()

    # Create some tools for the technical specialist
    def check_system_status(service_name: str) -> str:
        """Check the operational status of a service"""
        # Simulated system check
        statuses = {
            "api": "operational",
            "database": "degraded - investigating high latency",
            "auth": "operational",
            "payments": "operational"
        }
        return statuses.get(service_name.lower(), "unknown service")

    def restart_service(service_name: str) -> str:
        """Restart a service (requires elevated permissions)"""
        return f"Service '{service_name}' restart initiated. ETA: 2-3 minutes."

    tech_toolbox = Toolbox()
    tech_toolbox.add_tool(
        Tool.from_function(check_system_status)
    )
    tech_toolbox.add_tool(
        Tool.from_function(restart_service)
    )

    # Create specialized agents
    # 1. General Support Agent (first line)
    general_support = HandoffCaptain(
        provider=provider,
        model="gpt-4o-mini",
        system_prompt=(
            "You are a friendly General Support Agent. Handle common questions about "
            "account management, basic troubleshooting, and general inquiries. "
            "If the issue is technical (API errors, system issues, integrations), "
            "use the 'escalate_to_technical' tool. "
            "If the issue is about billing, pricing, or payments, use 'escalate_to_billing'. "
            "Always be friendly and explain why you're transferring them."
        ),
        name="GeneralSupport",
        color="cyan"
    )

    # 2. Technical Support Specialist
    technical_support = HandoffCaptain(
        provider=provider,
        model="gpt-4o-mini",
        instruments=tech_toolbox.get_tools(),
        system_prompt=(
            "You are a Technical Support Specialist. You handle complex technical issues, "
            "API problems, integration help, and system troubleshooting. "
            "Use your tools to check system status and restart services when needed. "
            "Provide detailed technical explanations. "
            "If the issue requires engineering review, use 'escalate_to_engineering'."
        ),
        name="TechnicalSupport",
        color="blue"
    )

    # 3. Billing Support Specialist
    billing_support = HandoffCaptain(
        provider=provider,
        model="gpt-4o-mini",
        system_prompt=(
            "You are a Billing Support Specialist. You handle questions about invoices, "
            "pricing, payment methods, refunds, and subscription management. "
            "Be empathetic with billing concerns and provide clear explanations of charges. "
            "You have access to customer billing history."
        ),
        name="BillingSupport",
        color="green"
    )

    # 4. Engineering Team (final escalation)
    engineering_team = HandoffCaptain(
        provider=provider,
        model="gpt-4o-mini",
        system_prompt=(
            "You are a Senior Engineer. You handle complex bugs, architectural questions, "
            "and issues that require code-level investigation. "
            "Acknowledge the issue, explain next steps, and set expectations for resolution timeline."
        ),
        name="Engineering",
        color="magenta"
    )

    # Register handoffs (creating the escalation chain)
    print("🔗 Setting up escalation chain:")
    print("   GeneralSupport → TechnicalSupport → Engineering")
    print("   GeneralSupport → BillingSupport")
    print()

    general_support.register_handoff(
        target_name="TechnicalSupport",
        target_captain=technical_support,
        handoff_description="Escalate technical issues to technical support specialist"
    )

    general_support.register_handoff(
        target_name="BillingSupport",
        target_captain=billing_support,
        handoff_description="Escalate billing and payment questions to billing specialist"
    )

    technical_support.register_handoff(
        target_name="Engineering",
        target_captain=engineering_team,
        handoff_description="Escalate complex bugs or architectural issues to engineering team"
    )

    # Scenario 1: Technical issue requiring escalation
    print("=" * 70)
    print("SCENARIO 1: Technical Issue with Escalation")
    print("=" * 70)
    print()

    technical_query = (
        "Hi, I'm getting 502 errors when calling your API endpoint /v1/users. "
        "This started about 30 minutes ago. Is there a system outage?"
    )

    print(f"Customer: {technical_query}")
    print()

    result1 = general_support.chat(
        message=technical_query,
        temperature=0.7,
        max_tokens=500,
        max_handoffs=3  # Allow up to 3 handoffs
    )

    print()
    print("Final Response:")
    print(result1.content)
    print()

    # Show handoff chain
    if "handoff_chain" in result1.metadata:
        chain = result1.metadata["handoff_chain"]
        print("🔄 Handoff Chain:")
        for i, agent in enumerate(chain):
            print(f"   {i + 1}. {agent}")
        print()

    # Scenario 2: Billing question
    print("=" * 70)
    print("SCENARIO 2: Billing Question")
    print("=" * 70)
    print()

    billing_query = (
        "I was charged $299 yesterday but I thought my plan was $199/month. "
        "Can you explain this charge?"
    )

    print(f"Customer: {billing_query}")
    print()

    # Reset conversation for new scenario
    general_support.clear_messages()

    result2 = general_support.chat(
        message=billing_query,
        temperature=0.7,
        max_tokens=500,
        max_handoffs=2
    )

    print()
    print("Final Response:")
    print(result2.content)
    print()

    if "handoff_chain" in result2.metadata:
        chain = result2.metadata["handoff_chain"]
        print("🔄 Handoff Chain:")
        for i, agent in enumerate(chain):
            print(f"   {i + 1}. {agent}")
        print()

    # Scenario 3: Simple question (no handoff needed)
    print("=" * 70)
    print("SCENARIO 3: Simple Question (No Handoff)")
    print("=" * 70)
    print()

    simple_query = "How do I reset my password?"

    print(f"Customer: {simple_query}")
    print()

    general_support.clear_messages()

    result3 = general_support.chat(
        message=simple_query,
        temperature=0.7,
        max_tokens=300,
        max_handoffs=2
    )

    print()
    print("Final Response:")
    print(result3.content)
    print()

    if "handoff_chain" in result3.metadata:
        chain = result3.metadata["handoff_chain"]
        if len(chain) == 1:
            print("✅ No handoff needed - handled by GeneralSupport")
        print()

    # Summary
    print("=" * 70)
    print("WHEN TO USE WATCH CHANGE PATTERN")
    print("=" * 70)
    print("""
    ✅ Use WatchChange When:
    - Clear escalation or handoff paths exist
    - Specialist consultation needed
    - Context must be preserved across transfers
    - Progressive specialization required
    - Support tiers or expertise levels

    ❌ Avoid WatchChange When:
    - Single agent can handle all cases
    - No clear handoff criteria
    - Context doesn't need preservation
    - Handoff overhead not justified

    💡 Tips:
    - Define clear handoff conditions
    - Use descriptive handoff names
    - Set reasonable max_handoffs (2-4)
    - Give each agent clear scope and expertise
    - Use color coding for visual distinction
    - Track handoff_chain in metadata

    🔗 Handoff Chain Tracking:
    - result.metadata["handoff_chain"] shows full path
    - result.metadata["handoff_history"] shows detailed transfer log
    - Useful for analytics and optimization
    """)

    # Show token usage
    print("=" * 70)
    print("RESOURCE USAGE")
    print("=" * 70)
    print(f"Scenario 1 (Technical): {result1.total_tokens:,} tokens")
    print(f"Scenario 2 (Billing): {result2.total_tokens:,} tokens")
    print(f"Scenario 3 (Simple): {result3.total_tokens:,} tokens")
    print()
    print("Note: Context is preserved across handoffs,")
    print("making each subsequent agent fully informed.")
    print()


if __name__ == "__main__":
    main()
