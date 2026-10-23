"""A τ-bench agent whose tool access is governed by a concordat protocol.

It is τ-bench's standard ``llm_agent`` with one difference: each turn, the model is
offered only the tools the protocol allows. The protocol encodes the rule common to
the τ-bench policies: anything that changes the database (a WRITE tool) needs the
customer's explicit "yes" first. Lookups are always available, and a transfer to a
human ends the session.

Run it through ``run.py``, which registers it as ``concordat_agent``.
"""

from __future__ import annotations

import re
from typing import Any

from tau2.agent.llm_agent import LLMAgent, LLMAgentState
from tau2.data_model.message import AssistantMessage, MultiToolMessage, UserMessage
from tau2.environment.tool import Tool
from tau2.environment.toolkit import TOOL_TYPE_ATTR, ToolType
from tau2.utils.llm_utils import generate

from concordat import Protocol, Session

TRANSFER = "transfer_to_human_agents"
YES = re.compile(r"\b(yes|yep|yeah)\b", re.IGNORECASE)  # the policies ask for an explicit "yes"

PROTOCOL_NOTE = """
<process>
Tools that change the database (bookings, cancellations, changes, refunds) are only
offered right after the customer has explicitly said yes to the exact action you
described. Until then you only have lookup tools: describe the action and ask the
customer to confirm. Earlier tool results stay valid when the tool list changes.
</process>
""".strip()


def protocol_for(tools: list[Tool]) -> Protocol:
    """chatting --user says yes--> confirmed; writes only in confirmed; transfer ends."""
    reads = {t.name for t in tools if _type(t) != ToolType.WRITE and t.name != TRANSFER}
    writes = {t.name for t in tools if _type(t) == ToolType.WRITE}
    chatting: dict[str, set[str]] = {name: {"chatting"} for name in reads}
    confirmed: dict[str, set[str]] = {name: {"confirmed"} for name in reads | writes}
    for actions in (chatting, confirmed):
        actions["user.yes"] = {"confirmed"}
        actions["user.other"] = {"chatting"}
        actions[TRANSFER] = {"transferred"}
    return Protocol(
        "chatting", {"chatting": chatting, "confirmed": confirmed}, final={"transferred"}
    )


def _type(tool: Tool) -> ToolType:
    return getattr(tool._func, TOOL_TYPE_ATTR, ToolType.READ)


class ConcordatAgent(LLMAgent[LLMAgentState]):
    def __init__(self, tools: list[Tool], domain_policy: str, llm: str, llm_args: Any = None):
        super().__init__(tools=tools, domain_policy=domain_policy, llm=llm, llm_args=llm_args)
        self.protocol = protocol_for(tools)
        self.session = Session(self.protocol)
        self.blocked: list[str] = []  # calls the model attempted but the protocol refused

    @property
    def system_prompt(self) -> str:
        return f"{super().system_prompt}\n{PROTOCOL_NOTE}"

    def _generate_next_message(self, message: Any, state: LLMAgentState) -> AssistantMessage:
        if isinstance(message, UserMessage) and not self.session.done:
            said_yes = bool(message.content and YES.search(message.content))
            self.session.step("user.yes" if said_yes else "user.other")
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)

        allowed = self.session.allowed_actions()
        tools = [t for t in self.tools if t.name in allowed]
        # The API needs a tool list once the history has tool calls, so with nothing
        # allowed (after a transfer) send the list but forbid calling any of it.
        response: AssistantMessage = generate(
            model=self.llm,
            tools=tools or self.tools,
            tool_choice=None if tools else "none",
            messages=state.system_messages + state.messages,
            call_name="agent_response",
            **self.llm_args,
        )
        calls = response.tool_calls or []
        if refused := [c.name for c in calls if c.name not in allowed]:
            # Never let a refused call reach the environment: ask the customer instead.
            self.blocked.extend(refused)
            return AssistantMessage(
                role="assistant",
                content="Before I make that change, can you confirm you'd like me to go ahead?",
            )
        for call in calls:
            self.session.step(call.name)
        return response


def create_concordat_agent(tools: list[Tool], domain_policy: str, **kwargs: Any) -> ConcordatAgent:
    return ConcordatAgent(
        tools=tools,
        domain_policy=domain_policy,
        llm=kwargs.get("llm"),
        llm_args=kwargs.get("llm_args"),
    )
