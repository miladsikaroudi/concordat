"""Run an agent against Claude through the Anthropic SDK (``pip install concordat[anthropic]``)."""

from __future__ import annotations

import inspect
import typing
from dataclasses import dataclass
from typing import Any, Literal

import anthropic
import pydantic
import pydantic_core

from .agent import Action, Agent
from .budget import Budget

DEFAULT_MODEL = "claude-opus-5-5"
FALLBACK_BETA = "server-side-fallback-2026-07-01"

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


def run(
    agent: Agent,
    client: anthropic.Anthropic,
    prompt: str | list[Any],
    *,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 16000,
    max_turns: int = 20,
    fallbacks: bool = True,
    **request: Any,
) -> RunResult:
    """Drive ``agent`` with Claude until its protocol finishes or a limit is hit.

    Each turn, Claude is offered only the actions legal in the agent's current state,
    and every response's tokens are charged to ``agent.budget``. ``fallbacks`` re-runs
    refused requests on Anthropic's recommended fallback model (a beta feature; turn it
    off on platforms without server-side fallbacks). Extra keyword arguments are passed
    to ``messages.create``.
    """
    messages = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else list(prompt)
    text = ""
    for _ in range(max_turns):
        if agent.done:
            return RunResult("done", text, messages)
        budget = agent.budget
        if _exhausted(budget):
            return RunResult("out_of_budget", text, messages)

        actions = agent.allowed_tools()
        params: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens if budget.tokens is None else min(max_tokens, budget.tokens),
            "messages": messages,
            **request,
        }
        if agent.instructions:
            params["system"] = agent.instructions
        if actions:
            params["tools"] = [_tool_param(action) for action in actions.values()]
            # One call per response, so each call sees the state the previous one left.
            params["tool_choice"] = {"type": "auto", "disable_parallel_tool_use": True}
        if fallbacks:
            response: Any = client.beta.messages.create(
                betas=[FALLBACK_BETA], fallbacks="default", **params
            )
        else:
            response = client.messages.create(**params)

        if response.stop_reason == "refusal":
            return RunResult("refused", text, messages)
        messages.append({"role": "assistant", "content": response.content})
        text = "".join(b.text for b in response.content if b.type == "text") or text
        calls = [b for b in response.content if b.type == "tool_use"]

        stop: Status | None = None
        if not _charge(budget, response.usage):
            stop = "out_of_budget"
        elif response.stop_reason == "max_tokens":
            stop = "max_tokens"
        if stop is not None:
            if calls:  # every tool_use needs a result, or the conversation can't continue
                results = [_result(c.id, f"Not run: {stop}.", error=True) for c in calls]
                messages.append({"role": "user", "content": results})
            return RunResult(stop, text, messages)
        if not calls:
            return RunResult("done" if agent.done else "waiting", text, messages)

        results = [_result(c.id, *_invoke(actions, c.name, c.input)) for c in calls]
        messages.append({"role": "user", "content": results})
    return RunResult("done" if agent.done else "max_turns", text, messages)


def _exhausted(budget: Budget) -> bool:
    return (budget.tokens is not None and budget.tokens <= 0) or (
        budget.tool_calls is not None and budget.tool_calls <= 0
    )


def _charge(budget: Budget, usage: Any) -> bool:
    """Charge all tokens the request processed; on overspend, empty the budget instead."""
    used = (
        usage.input_tokens
        + usage.output_tokens
        + (usage.cache_creation_input_tokens or 0)
        + (usage.cache_read_input_tokens or 0)
    )
    if budget.tokens is not None and used > budget.tokens:
        budget.tokens = 0
        return False
    budget.spend(tokens=used)
    return True


def _arguments(action: Action) -> type[pydantic.BaseModel]:
    hints = typing.get_type_hints(action.call)
    fields: dict[str, Any] = {
        name: (hints.get(name, Any), ... if p.default is p.empty else p.default)
        for name, p in inspect.signature(action.call).parameters.items()
    }
    return pydantic.create_model(f"{action.name}_arguments", **fields)


def _tool_param(action: Action) -> dict[str, Any]:
    description = action.description
    if action.human:
        description = (
            "Ask a human to decide; returns 'approved' or 'rejected'. " + description
        ).strip()
    return {
        "name": action.name,
        "description": description,
        "input_schema": _arguments(action).model_json_schema(),
    }


def _invoke(actions: dict[str, Action], name: str, raw: Any) -> tuple[str, bool]:
    """Validate and run one call. Returns the result text and whether it is an error."""
    action = actions.get(name)
    if action is None:
        return f"{name!r} is not allowed in the current state.", True
    try:
        arguments = _arguments(action).model_validate(raw)
        result = action.call(**{f: getattr(arguments, f) for f in type(arguments).model_fields})
    except Exception as e:  # the model sees the error and can correct itself
        return f"{type(e).__name__}: {e}", True
    if action.human:
        return ("approved" if result else "rejected"), False
    return (result if isinstance(result, str) else pydantic_core.to_json(result).decode()), False


def _result(tool_use_id: str, content: str, error: bool = False) -> dict[str, Any]:
    return {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": content,
        "is_error": error,
    }
