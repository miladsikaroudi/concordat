"""A refund agent driven by Claude. Run: uv run --extra anthropic python examples/refund.py"""

import anthropic

from concordat import Agent, Budget, cli_approver, human, tool
from concordat.anthropic import run

ORDERS = {"A-100": 42.0}


class RefundAgent(Agent):
    instructions = "You handle refund requests for an online shop. Be brief."
    final = {"refunded", "denied"}
    budget = Budget(tokens=60_000, tool_calls=6)

    @tool(when="start", then={True: "verified", False: "denied"})
    def verify_order(self, order_id: str) -> bool:
        """Check that the order exists. Must happen before any refund."""
        return order_id in ORDERS

    @tool(when="verified", then="awaiting_approval")
    def propose_refund(self, order_id: str, amount: float) -> str:
        """Propose a refund amount for a verified order."""
        return f"Proposed {amount} for {order_id} (order total {ORDERS[order_id]})."

    @human(when="awaiting_approval", approve="refunded", reject="denied")
    def approve_refund(self, order_id: str, amount: float) -> str:
        """Get a human's approval for the proposed refund."""
        return f"Refund {amount} for order {order_id}?"


if __name__ == "__main__":
    agent = RefundAgent(approver=cli_approver)
    result = run(agent, anthropic.Anthropic(), "Please refund order A-100 in full.")
    print(f"\n{result.status} in state {agent.state!r}: {result.text}")
    print(f"budget left: {agent.budget}")
