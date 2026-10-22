# FakeClient stands in for openai.OpenAI, so arguments typed as the client mismatch.
# mypy: disable-error-code="arg-type"
import json
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from concordat import Agent, Budget, human, tool
from concordat.openai import run

MODEL = "some-model"


@dataclass
class Proposal:
    amount: float


class RefundAgent(Agent):
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
    return SimpleNamespace(
        type="function_call", call_id=f"call_{name}", name=name, arguments=json.dumps(arguments)
    )


def say(text: str) -> Any:
    return SimpleNamespace(type="message", content=[SimpleNamespace(type="output_text", text=text)])


def refuse() -> Any:
    return SimpleNamespace(type="message", content=[SimpleNamespace(type="refusal", refusal="no")])


def reply(*items: Any, tokens: int = 15, incomplete: str | None = None) -> Any:
    text = "".join(
        p.text for i in items if i.type == "message" for p in i.content if p.type == "output_text"
    )
    return SimpleNamespace(
        output=list(items),
        output_text=text,
        usage=SimpleNamespace(total_tokens=tokens),
        status="incomplete" if incomplete else "completed",
        incomplete_details=SimpleNamespace(reason=incomplete) if incomplete else None,
    )


class FakeClient:
    """Plays back scripted responses and records each request."""

    def __init__(self, *scripted: Any) -> None:
        self.scripted = list(scripted)
        self.requests: list[dict[str, Any]] = []
        self.responses = SimpleNamespace(create=self._create)

    def _create(self, **params: Any) -> Any:
        self.requests.append({**params, "input": list(params["input"])})
        return self.scripted.pop(0)

    def offered(self) -> list[list[str]]:
        return [[t["name"] for t in r.get("tools", [])] for r in self.requests]


fake = FakeClient


def outputs(items: list[Any]) -> list[str]:
    return [
        i["output"]
        for i in items
        if isinstance(i, dict) and i.get("type") == "function_call_output"
    ]


def test_full_run_offers_only_legal_actions_each_turn() -> None:
    client = fake(
        reply(call("verify_identity", user_id="u1")),
        reply(say("Proposing."), call("propose_refund", amount=42)),
        reply(call("approve_refund", proposal={"amount": 42})),
    )
    agent = RefundAgent(approver=lambda step, request: request == "Refund 42.0?")
    result = run(agent, client, "Refund my order, I'm u1", model=MODEL)

    assert (result.status, agent.state, result.text) == ("done", "refunded", "Proposing.")
    assert client.offered() == [["verify_identity"], ["propose_refund"], ["approve_refund"]]
    assert outputs(result.messages) == ["true", '{"amount":42.0}', "approved"]
    assert agent.budget.tool_calls == 3

    first = client.requests[0]
    assert (first["model"], first["instructions"]) == (MODEL, "You process refunds.")
    assert (first["tool_choice"], first["parallel_tool_calls"]) == ("auto", False)
    [verify] = first["tools"]
    assert (verify["type"], verify["name"]) == ("function", "verify_identity")
    assert verify["parameters"]["required"] == ["user_id"]


def test_illegal_or_invalid_calls_are_reported_back_to_the_model() -> None:
    client = fake(
        reply(call("propose_refund", amount=10**6)),
        reply(call("verify_identity")),
        reply(say("Sorry.")),
    )
    agent = RefundAgent(approver=lambda s, r: True)
    result = run(agent, client, "hi", model=MODEL)

    illegal, invalid = outputs(result.messages)
    assert illegal.startswith("Error:") and "not allowed" in illegal
    assert invalid.startswith("Error: ValidationError")
    assert (result.status, agent.state) == ("waiting", "start")


def test_tokens_are_charged_and_cap_the_next_response() -> None:
    class Metered(RefundAgent):
        budget = Budget(tokens=100)

    client = fake(
        reply(call("verify_identity", user_id="u1"), tokens=90),
        reply(call("propose_refund", amount=1), tokens=50),
    )
    agent = Metered(approver=lambda s, r: True)
    result = run(agent, client, "hi", model=MODEL)

    assert client.requests[1]["max_output_tokens"] == 10
    assert (result.status, agent.budget.tokens, agent.state) == ("out_of_budget", 0, "verified")
    assert outputs(result.messages)[-1] == "Error: Not run: out_of_budget."


def test_truncated_response_stops_without_running_calls() -> None:
    client = fake(reply(call("verify_identity", user_id="u1"), incomplete="max_output_tokens"))
    agent = RefundAgent(approver=lambda s, r: True)
    result = run(agent, client, "hi", model=MODEL)
    assert (result.status, agent.state) == ("max_tokens", "start")


def test_refusals() -> None:
    for response in (reply(refuse()), reply(incomplete="content_filter")):
        result = run(RefundAgent(approver=lambda s, r: True), fake(response), "hi", model=MODEL)
        assert result.status == "refused"
        assert len(result.messages) == 1
