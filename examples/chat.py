"""A terminal chat with memory, injected dependencies, a handoff, and a human approving risky actions.

export ANTHROPIC_API_KEY=...
python examples/chat.py
"""

import asyncio
from dataclasses import dataclass, field

from anthropic import AsyncAnthropic

from fleet import Agent, Context, Handoff, Message, TextDelta, ToolCall


@dataclass
class Shop:
    """Stands in for your database. Tools reach it through the run context, not a global."""

    customer: str
    orders: dict[str, float] = field(default_factory=lambda: {"A100": 24.50, "A101": 310.00})
    refunded: list[str] = field(default_factory=list)


def list_orders(ctx: Context[Shop]) -> dict[str, float]:
    """List the customer's orders and what each cost."""
    return ctx.deps.orders


def refund(ctx: Context[Shop], order_id: str) -> str:
    """Refund an order in full.

    Args:
        order_id: The order to refund, such as "A100".
    """
    if order_id not in ctx.deps.orders:
        raise ValueError(f"No order {order_id}.")
    ctx.deps.refunded.append(order_id)
    return f"Refunded {order_id}."


# Passed straight to the Messages API: if the model declines a request, let the API retry it on another model.
CLAUDE = {"betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"}

billing = Agent(
    name="billing",
    description="Handles refunds and questions about charges.",
    instructions=lambda ctx: (
        f"You handle billing for {ctx.deps.customer}. Check their orders before refunding anything."
    ),
    tools=[list_orders, refund],
    params=CLAUDE,
)
front_desk = Agent(
    name="front desk",
    instructions="Greet the customer and answer general questions. Hand billing matters to billing.",
    handoffs=[billing],
    params=CLAUDE,
)


async def approve(ctx: Context[Shop], call: ToolCall) -> bool | str:
    """Ask the person at the keyboard before money moves."""
    if call.name != "refund":
        return True
    answer = await asyncio.to_thread(input, f"\n  Allow refund of {call.arguments['order_id']}? [y/N] ")
    return answer.strip().lower() == "y" or "The operator declined this refund."


async def main() -> None:
    client = AsyncAnthropic()
    shop = Shop(customer="Ada")
    agent, history = front_desk, list[Message]()

    def show(event: object) -> None:
        if isinstance(event, TextDelta):
            print(event.text, end="", flush=True)
        elif isinstance(event, Handoff):
            print(f"  (transferred to {event.target})")

    while (text := input("\nyou> ").strip()) not in ("", "quit", "exit"):
        result = await agent.run(
            text,
            history=history,
            deps=shop,
            client=client,
            model="claude-opus-5-5",
            on_event=show,
            approve=approve,
        )
        # Keep talking to whichever agent ended up answering.
        agent, history = result.agent, result.messages
        print()


if __name__ == "__main__":
    asyncio.run(main())
