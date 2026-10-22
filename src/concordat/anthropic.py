"""Run an agent against Claude through the Anthropic SDK (``pip install concordat[anthropic]``)."""

from __future__ import annotations

from typing import Any

import anthropic

from ._runner import RunResult, Status, arguments, cap, charge, description, exhausted, invoke
from .agent import Action, Agent

__all__ = ["DEFAULT_MODEL", "FALLBACK_BETA", "RunResult", "run"]

DEFAULT_MODEL = "claude-opus-5-5"
FALLBACK_BETA = "server-side-fallback-2026-07-01"


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
        if exhausted(budget):
            return RunResult("out_of_budget", text, messages)

        actions = agent.allowed_tools()
        params: dict[str, Any] = {
            "model": model,
            "max_tokens": cap(budget, max_tokens),
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
        if not charge(budget, _tokens(response.usage)):
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

        results = [_result(c.id, *invoke(actions, c.name, c.input)) for c in calls]
        messages.append({"role": "user", "content": results})
    return RunResult("done" if agent.done else "max_turns", text, messages)


def _tokens(usage: Any) -> int:
    return int(
        usage.input_tokens
        + usage.output_tokens
        + (usage.cache_creation_input_tokens or 0)
        + (usage.cache_read_input_tokens or 0)
    )


def _tool_param(action: Action) -> dict[str, Any]:
    return {
        "name": action.name,
        "description": description(action),
        "input_schema": arguments(action).model_json_schema(),
    }


def _result(tool_use_id: str, content: str, error: bool = False) -> dict[str, Any]:
    return {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": content,
        "is_error": error,
    }
