"""Offline tests for the τ-bench adapter: no model calls, using the real airline tools.

uv run --project tau2-bench pytest test_concordat_tau.py
"""

from typing import Any

import concordat_tau
import pytest
from concordat_tau import TRANSFER, ConcordatAgent
from tau2.data_model.message import AssistantMessage, ToolCall, ToolMessage, UserMessage
from tau2.registry import registry

AIRLINE = registry.get_env_constructor("airline")()


def agent_with(*responses: AssistantMessage, offered: list[list[str]]) -> ConcordatAgent:
    scripted = list(responses)

    def fake_generate(*, tools: Any, **_: Any) -> AssistantMessage:
        offered.append(sorted(t.name for t in tools or []))
        return scripted.pop(0)

    concordat_tau.generate = fake_generate  # type: ignore[assignment]
    return ConcordatAgent(AIRLINE.get_tools(), AIRLINE.get_policy(), llm="fake")


def call(name: str, **arguments: Any) -> AssistantMessage:
    return AssistantMessage(
        role="assistant", tool_calls=[ToolCall(id=name, name=name, arguments=arguments)]
    )


def say(text: str) -> AssistantMessage:
    return AssistantMessage(role="assistant", content=text)


def user(text: str) -> UserMessage:
    return UserMessage(role="user", content=text)


def result(name: str) -> ToolMessage:
    return ToolMessage(id=name, role="tool", content="ok")


def test_writes_are_offered_only_after_an_explicit_yes() -> None:
    offered: list[list[str]] = []
    agent = agent_with(
        call("get_reservation_details", reservation_id="R1"),
        say("Shall I cancel R1?"),
        call("cancel_reservation", reservation_id="R1"),
        offered=offered,
    )
    state = agent.get_init_state()
    agent.generate_next_message(user("Cancel reservation R1 please"), state)
    agent.generate_next_message(result("get_reservation_details"), state)
    assert "cancel_reservation" not in offered[0] and "cancel_reservation" not in offered[1]
    assert "get_reservation_details" in offered[0] and TRANSFER in offered[0]

    response, _ = agent.generate_next_message(user("Yes, go ahead."), state)
    assert "cancel_reservation" in offered[2]
    assert response.tool_calls and response.tool_calls[0].name == "cancel_reservation"
    assert agent.session.state == "confirmed"


def test_a_write_the_model_attempts_without_a_yes_never_reaches_the_environment() -> None:
    agent = agent_with(call("cancel_reservation", reservation_id="R1"), offered=[])
    response, _ = agent.generate_next_message(user("Cancel R1"), agent.get_init_state())
    assert response.tool_calls is None
    assert "confirm" in (response.content or "")
    assert agent.blocked == ["cancel_reservation"]


def test_a_new_request_withdraws_the_confirmation() -> None:
    agent = agent_with(say("ok"), say("Which flight?"), offered=[])
    state = agent.get_init_state()
    agent.generate_next_message(user("yes"), state)
    assert agent.session.state == "confirmed"
    agent.generate_next_message(user("Actually, change my flight instead"), state)
    assert agent.session.state == "chatting"


def test_transfer_ends_the_session() -> None:
    agent = agent_with(call(TRANSFER, summary="needs a human"), say("Transferring."), offered=[])
    state = agent.get_init_state()
    agent.generate_next_message(user("I want a human"), state)
    assert agent.session.done
    # Afterwards tools are still sent (the API requires them) but none may be called.
    calls: list[Any] = []
    concordat_tau.generate = lambda **kw: calls.append(kw) or say("Bye.")  # type: ignore[assignment]
    agent.generate_next_message(result(TRANSFER), state)
    assert calls[0]["tool_choice"] == "none" and calls[0]["tools"]


@pytest.mark.parametrize("domain", ["airline", "retail", "telecom"])
def test_protocol_is_valid_for_each_domain(domain: str) -> None:
    env = registry.get_env_constructor(domain)()
    protocol = concordat_tau.protocol_for(env.get_tools())
    assert protocol.actions("chatting")  # building it ran concordat's checks


def test_pushback_is_not_a_confirmation() -> None:
    agent = agent_with(say("Please confirm."), offered=[])
    agent.generate_next_message(
        user("I already told you what I want, just do it and go ahead, stop asking."),
        agent.get_init_state(),
    )
    assert agent.session.state == "chatting"


def test_pressure_task_set_adds_the_pushback_instruction() -> None:
    import run

    plain = registry.get_tasks_loader("airline")()
    pressured = run.pressure_tasks()
    assert [t.id for t in pressured] == [t.id for t in plain]
    instructions = pressured[0].user_scenario.instructions.task_instructions
    assert instructions.endswith(run.PRESSURE)
