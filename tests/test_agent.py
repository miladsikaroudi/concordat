from dataclasses import dataclass

import pytest

from concordat import (
    Agent,
    Budget,
    BudgetExceeded,
    ProtocolError,
    ProtocolViolation,
    human,
    tool,
)


@dataclass
class Proposal:
    amount: float


class RefundAgent(Agent):
    final = {"refunded", "denied"}
    budget = Budget(tool_calls=5)

    def __init__(self, approve: bool = True) -> None:
        super().__init__(approver=lambda step, request: approve)
        self.calls: list[str] = []

    @tool(when="start", then={True: "verified", False: "denied"})
    def verify_identity(self, user_id: str) -> bool:
        self.calls.append("verify_identity")
        return user_id == "u1"

    @tool(when="verified", then="awaiting_approval")
    def propose_refund(self, amount: float) -> Proposal:
        self.calls.append("propose_refund")
        return Proposal(amount)

    @human(when="awaiting_approval", approve="refunded", reject="denied")
    def approve_refund(self, proposal: Proposal) -> str:
        return f"Refund {proposal.amount}?"


def test_protocol_is_built_from_decorators() -> None:
    p = RefundAgent.protocol
    assert p.actions("start") == {"verify_identity"}
    assert p.targets("start", "verify_identity") == {"verified", "denied"}
    assert p.targets("awaiting_approval", "approve_refund") == {"refunded", "denied"}


def test_happy_path() -> None:
    agent = RefundAgent()
    assert agent.verify_identity("u1") is True
    proposal = agent.propose_refund(42.0)
    assert agent.state == "awaiting_approval"
    assert agent.approve_refund(proposal) is True
    assert agent.done and agent.state == "refunded"


def test_result_selects_the_branch() -> None:
    agent = RefundAgent()
    agent.verify_identity("someone else")
    assert agent.done and agent.state == "denied"


def test_human_rejection() -> None:
    agent = RefundAgent(approve=False)
    agent.verify_identity("u1")
    assert agent.approve_refund(agent.propose_refund(42.0)) is False
    assert agent.state == "denied"


def test_out_of_order_call_is_rejected_before_it_runs() -> None:
    agent = RefundAgent()
    with pytest.raises(ProtocolViolation, match="not allowed in state 'start'"):
        agent.propose_refund(1_000_000.0)
    assert agent.calls == []
    assert agent.state == "start"


def test_allowed_tools_follow_the_state() -> None:
    agent = RefundAgent()
    assert list(agent.allowed_tools()) == ["verify_identity"]
    agent.verify_identity("u1")
    assert list(agent.allowed_tools()) == ["propose_refund"]
    agent.propose_refund(10.0)
    [request] = agent.allowed_tools().values()
    assert (request.name, request.human) == ("approve_refund", True)


def test_bound_methods_drop_self_from_their_signature() -> None:
    import inspect

    action = RefundAgent().allowed_tools()["verify_identity"]
    assert list(inspect.signature(action.call).parameters) == ["user_id"]


def test_tool_calls_are_charged_per_instance() -> None:
    agent = RefundAgent()
    agent.verify_identity("u1")
    agent.propose_refund(1.0)
    assert agent.budget.tool_calls == 3
    assert RefundAgent.budget.tool_calls == 5
    assert RefundAgent().budget.tool_calls == 5


def test_exhausted_budget_blocks_the_call() -> None:
    class Tiny(RefundAgent):
        budget = Budget(tool_calls=1)

    agent = Tiny()
    agent.verify_identity("u1")
    with pytest.raises(BudgetExceeded):
        agent.propose_refund(1.0)
    assert agent.calls == ["verify_identity"]


def test_unmapped_result_is_a_violation() -> None:
    class Flaky(Agent):
        final = {"done"}

        @tool(when="start", then={"ok": "done"})
        def check(self) -> str:
            return "maybe"

    agent = Flaky(approver=lambda step, request: True)
    with pytest.raises(ProtocolViolation, match="returned 'maybe'"):
        agent.check()
    assert agent.state == "start"


def test_bad_protocol_fails_at_class_definition() -> None:
    with pytest.raises(ProtocolError, match="Stuck: states that can never finish"):

        class Stuck(Agent):
            final = {"done"}

            @tool(when="start", then={True: "done", False: "limbo"})
            def go(self) -> bool:
                return True


def test_abstract_base_and_inheritance() -> None:
    class Support(Agent):  # no final states: an abstract base
        @tool(when="start", then="answered")
        def answer(self, question: str) -> str:
            return "42"

    with pytest.raises(TypeError, match="abstract"):
        Support()

    class EscalatingSupport(Support):
        final = {"closed"}

        @human(when="answered", approve="closed", reject="start")
        def satisfied(self) -> str:
            return "Is the customer happy?"

    agent = EscalatingSupport(approver=lambda step, request: False)
    agent.answer("meaning of life?")
    agent.satisfied()
    assert agent.state == "start"  # rejected: back to answering
