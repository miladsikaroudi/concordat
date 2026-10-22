"""Provider-neutral parts of the model runners. Needs pydantic, which every provider SDK ships."""

from __future__ import annotations

import inspect
import typing
from dataclasses import dataclass
from typing import Any, Literal

import pydantic
import pydantic_core

from .agent import Action, Agent
from .budget import Budget

STEPWISE_NOTE = (
    "This conversation follows a fixed process. Each turn you are offered only the "
    "actions allowed at the current step, so tools you used earlier may no longer be "
    "listed and new ones may appear. That is expected: earlier results remain valid."
)

Status = Literal["done", "waiting", "refused", "max_tokens", "out_of_budget", "max_turns"]


@dataclass
class RunResult:
    """How a run ended.

    ``done``: the protocol reached a final state. ``waiting``: the model ended its turn
    before that, usually to ask the user something; append a user message to
    ``messages`` and call ``run`` again to continue.
    """

    status: Status
    text: str
    messages: list[Any]


def system_prompt(agent: Agent) -> str:
    return f"{agent.instructions}\n\n{STEPWISE_NOTE}" if agent.instructions else STEPWISE_NOTE


def with_next_step(agent: Agent, content: str) -> str:
    """Append what the model may do next, so a changing tool list is never a surprise."""
    if agent.done:
        return f"{content}\n\n[The process is complete.]"
    allowed = ", ".join(agent.allowed_tools())
    return f"{content}\n\n[Now allowed: {allowed}.]"


def exhausted(budget: Budget) -> bool:
    return (budget.tokens is not None and budget.tokens <= 0) or (
        budget.tool_calls is not None and budget.tool_calls <= 0
    )


def cap(budget: Budget, max_tokens: int) -> int:
    """The output limit for the next response: never more than the tokens left."""
    return max_tokens if budget.tokens is None else min(max_tokens, budget.tokens)


def charge(budget: Budget, used: int) -> bool:
    """Charge the tokens a request processed; on overspend, empty the budget instead."""
    if budget.tokens is not None and used > budget.tokens:
        budget.tokens = 0
        return False
    budget.spend(tokens=used)
    return True


def arguments(action: Action) -> type[pydantic.BaseModel]:
    """A model of the action's parameters, from its signature and type hints."""
    hints = typing.get_type_hints(action.call)
    fields: dict[str, Any] = {
        name: (hints.get(name, Any), ... if p.default is p.empty else p.default)
        for name, p in inspect.signature(action.call).parameters.items()
    }
    return pydantic.create_model(f"{action.name}_arguments", **fields)


def description(action: Action) -> str:
    if action.human:
        return (
            "Ask a human to decide; returns 'approved' or 'rejected'. " + action.description
        ).strip()
    return action.description


def invoke(actions: dict[str, Action], name: str, raw: Any) -> tuple[str, bool]:
    """Validate and run one call. Returns the result text and whether it is an error.

    ``raw`` is the call's arguments as a dict, or as a JSON string.
    """
    action = actions.get(name)
    if action is None:
        return f"{name!r} is not allowed in the current state.", True
    try:
        model = arguments(action)
        args = model.model_validate_json(raw) if isinstance(raw, str) else model.model_validate(raw)
        result = action.call(**{f: getattr(args, f) for f in type(args).model_fields})
    except Exception as e:  # the model sees the error and can correct itself
        return f"{type(e).__name__}: {e}", True
    if action.human:
        return ("approved" if result else "rejected"), False
    return (result if isinstance(result, str) else pydantic_core.to_json(result).decode()), False
