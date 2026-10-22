# FakeClient stands in for anthropic.Anthropic, so arguments typed as the client mismatch.
# mypy: disable-error-code="arg-type"
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from concordat import Agent, Budget, human, tool
from concordat.anthropic import FALLBACK_BETA, run


@dataclass
class Proposal:
    amount: float


class RefundAgent(Agent):
    """Handles refund requests."""

    instructions = "You process refunds."
    final = {"refunded", "denied"}
    budget = Budget(tool_calls=5)

    @tool(when="start", then={True: "verified", False: "denied"})
    def verify_identity(self, user_id: str) -> bool:
        """Check the customer's identity."""
        return user_id == "u1"

    @tool(when="verified", then="awaiting_approval")
    def propose_refund(self, amount: float) -> Proposal:
        return Proposal(amount)

    @human(when="awaiting_approval", approve="refunded", reject="denied")
    def approve_refund(self, proposal: Proposal) -> str:
        return f"Refund {proposal.amount}?"


def call(name: str, **arguments: Any) -> Any:
    return SimpleNamespace(type="tool_use", id=f"id_{name}", name=name, input=arguments)


def say(text: str) -> Any:
    return SimpleNamespace(type="text", text=text)


def reply(*blocks: Any, stop: str = "tool_use", tokens: tuple[int, int] = (10, 5)) -> Any:
    usage = SimpleNamespace(
        input_tokens=tokens[0],
        output_tokens=tokens[1],
        cache_creation_input_tokens=None,
        cache_read_input_tokens=0,
    )
    return SimpleNamespace(content=list(blocks), stop_reason=stop, usage=usage)


class FakeClient:
    """Plays back scripted responses and records each request."""

    def __init__(self, *responses: Any) -> None:
        self.responses = list(responses)
        self.requests: list[dict[str, Any]] = []
        self.messages = SimpleNamespace(create=self._create)
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **params: Any) -> Any:
        self.requests.append({**params, "messages": list(params["messages"])})
        return self.responses.pop(0)

    def offered(self) -> list[list[str]]:
        return [[t["name"] for t in r.get("tools", [])] for r in self.requests]


def last_results(messages: list[Any]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = messages[-1]["content"]
    return results


def test_full_run_offers_only_legal_actions_each_turn() -> None:
    client = FakeClient(
        reply(call("verify_identity", user_id="u1")),
        reply(say("Proposing."), call("propose_refund", amount=42)),
        reply(call("approve_refund", proposal={"amount": 42})),
    )
    agent = RefundAgent(approver=lambda step, request: request == "Refund 42.0?")
    result = run(agent, client, "Refund my order, I'm u1")

    assert result.status == "done"
    assert agent.state == "refunded"
    assert result.text == "Proposing."
    assert client.offered() == [["verify_identity"], ["propose_refund"], ["approve_refund"]]
    assert last_results(result.messages) == [
        {
            "type": "tool_result",
            "tool_use_id": "id_approve_refund",
            "content": "approved",
            "is_error": False,
        }
    ]
    assert agent.budget.tool_calls == 3  # two tool calls; asking a human is free

    first = client.requests[0]
    assert first["system"] == "You process refunds."
    assert first["model"] == "claude-opus-5-5"
    assert first["tool_choice"] == {"type": "auto", "disable_parallel_tool_use": True}
    assert (first["betas"], first["fallbacks"]) == ([FALLBACK_BETA], "default")
    [verify] = first["tools"]
    assert verify["description"] == "Check the customer's identity."
    assert verify["input_schema"]["required"] == ["user_id"]


def test_human_actions_are_labelled_for_the_model() -> None:
    client = FakeClient(
        reply(call("verify_identity", user_id="u1")),
        reply(call("propose_refund", amount=5)),
        reply(say("Asked."), stop="end_turn"),
    )
    run(RefundAgent(approver=lambda s, r: True), client, "hi")
    [approve] = client.requests[2]["tools"]
    assert approve["description"].startswith("Ask a human to decide")


def test_human_rejection_ends_in_denied() -> None:
    client = FakeClient(
        reply(call("verify_identity", user_id="u1")),
        reply(call("propose_refund", amount=5)),
        reply(call("approve_refund", proposal={"amount": 5})),
    )
    agent = RefundAgent(approver=lambda s, r: False)
    result = run(agent, client, "hi")
    assert (result.status, agent.state) == ("done", "denied")
    assert last_results(result.messages)[0]["content"] == "rejected"


def test_model_asking_the_user_returns_waiting() -> None:
    client = FakeClient(reply(say("What is your user id?"), stop="end_turn"))
    agent = RefundAgent(approver=lambda s, r: True)
    result = run(agent, client, "Refund please")
    assert (result.status, result.text, agent.state) == (
        "waiting",
        "What is your user id?",
        "start",
    )

    client.responses.append(reply(call("verify_identity", user_id="u2")))
    result = run(
        agent,
        client,
        [*result.messages, {"role": "user", "content": "u2"}],
    )
    assert (result.status, agent.state) == ("done", "denied")


def test_illegal_or_invalid_calls_are_reported_back_to_the_model() -> None:
    client = FakeClient(
        reply(call("propose_refund", amount=10**6)),  # not offered in this state
        reply(call("verify_identity")),  # missing argument
        reply(say("Sorry."), stop="end_turn"),
    )
    agent = RefundAgent(approver=lambda s, r: True)
    result = run(agent, client, "hi")

    illegal = client.requests[1]["messages"][-1]["content"][0]
    invalid = client.requests[2]["messages"][-1]["content"][0]
    assert illegal["is_error"] and "not allowed" in illegal["content"]
    assert invalid["is_error"] and "ValidationError" in invalid["content"]
    assert (result.status, agent.state) == ("waiting", "start")
    assert agent.budget.tool_calls == 5  # nothing ran


def test_tokens_are_charged_and_cap_the_next_response() -> None:
    class Metered(RefundAgent):
        budget = Budget(tokens=100)

    client = FakeClient(
        reply(call("verify_identity", user_id="u1"), tokens=(60, 30)),
        reply(call("propose_refund", amount=1), tokens=(40, 10)),
    )
    agent = Metered(approver=lambda s, r: True)
    result = run(agent, client, "hi")

    assert client.requests[1]["max_tokens"] == 10  # only 10 tokens were left
    assert result.status == "out_of_budget"
    assert agent.budget.tokens == 0
    assert agent.state == "verified"  # the over-budget call was not run
    assert last_results(result.messages)[0]["content"] == "Not run: out_of_budget."


def test_refusal_is_not_appended() -> None:
    client = FakeClient(reply(stop="refusal"))
    result = run(RefundAgent(approver=lambda s, r: True), client, "hi")
    assert result.status == "refused"
    assert len(result.messages) == 1


def test_fallbacks_can_be_turned_off() -> None:
    client = FakeClient(reply(say("ok"), stop="end_turn"))
    run(RefundAgent(approver=lambda s, r: True), client, "hi", fallbacks=False)
    assert "betas" not in client.requests[0] and "fallbacks" not in client.requests[0]


def test_max_turns() -> None:
    client = FakeClient(*[reply(call("propose_refund", amount=1)) for _ in range(3)])
    result = run(
        RefundAgent(approver=lambda s, r: True),
        client,
        "hi",
        max_turns=3,
    )
    assert result.status == "max_turns"
