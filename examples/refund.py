"""A refund agent driven by Claude. Run: uv run --extra anthropic python examples/refund.py"""

import anthropic

from concordat import Agent, Budget, cli_approver, human, tool
from concordat.anthropic import run

ORDERS = {"A-100": 42.0}


class RefundAgent(Agent):
    instructions = "You handle refund requests for an online shop. Be brief."
    final = {"refunded", "denied"}
    budget = Budget(tokens=60_000, tool_calls=6)

    @tool(when="start", then="verified")
    def look_up_order(self, order_id: str) -> dict[str, str | float]:
        """Find the order and its total. Must happen before any refund."""
        if order_id not in ORDERS:
            raise KeyError(f"no order {order_id!r}")  # an error leaves the state unchanged
        return {"order_id": order_id, "total": ORDERS[order_id]}

    @tool(when="verified", then="awaiting_approval")
    def propose_refund(self, order_id: str, amount: float) -> str:
        """Propose a refund amount, at most the order total."""
        if amount > ORDERS[order_id]:
            raise ValueError(f"amount exceeds the order total {ORDERS[order_id]}")
        return f"Proposed a refund of {amount} for {order_id}."

    @human(when="awaiting_approval", approve="refunded", reject="denied")
    def approve_refund(self, order_id: str, amount: float) -> str:
        """Get a human's approval for the proposed refund."""
        return f"Refund {amount} for order {order_id}?"


if __name__ == "__main__":
    agent = RefundAgent(approver=cli_approver)
    result = run(agent, anthropic.Anthropic(), "Please refund order A-100 in full.")
    print(f"\n{result.status} in state {agent.state!r}: {result.text}")
    print(f"budget left: {agent.budget}")
