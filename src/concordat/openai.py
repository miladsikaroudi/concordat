"""Run an agent through the OpenAI Responses API (``pip install concordat[openai]``)."""

from __future__ import annotations

from typing import Any

import openai

from ._runner import (
    RunResult,
    Status,
    arguments,
    cap,
    charge,
    description,
    exhausted,
    invoke,
    system_prompt,
    with_next_step,
)
from .agent import Action, Agent

__all__ = ["RunResult", "run"]


def run(
    agent: Agent,
    client: openai.OpenAI,
    prompt: str | list[Any],
    *,
    model: str,
    max_output_tokens: int = 16000,
    max_turns: int = 20,
    **request: Any,
) -> RunResult:
    """Drive ``agent`` with an OpenAI model until its protocol finishes or a limit is hit.

    Each turn, the model is offered only the actions legal in the agent's current state,
    and every response's tokens are charged to ``agent.budget``. ``messages`` in the
    result are Responses API input items. Extra keyword arguments are passed to
    ``responses.create``.
    """
    items = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else list(prompt)
    text = ""
    for _ in range(max_turns):
        if agent.done:
            return RunResult("done", text, items)
        budget = agent.budget
        if exhausted(budget):
            return RunResult("out_of_budget", text, items)

        actions = agent.allowed_tools()
        params: dict[str, Any] = {
            "model": model,
            "input": items,
            "max_output_tokens": cap(budget, max_output_tokens),
            "instructions": system_prompt(agent),
            **request,
        }
        if actions:
            params["tools"] = [_tool_param(action) for action in actions.values()]
            # One call per response, so each call sees the state the previous one left.
            params["tool_choice"] = "auto"
            params["parallel_tool_calls"] = False
        response: Any = client.responses.create(**params)

        if _refused(response):
            return RunResult("refused", text, items)
        items.extend(response.output)
        text = response.output_text or text
        calls = [item for item in response.output if item.type == "function_call"]

        stop: Status | None = None
        if not charge(budget, response.usage.total_tokens if response.usage else 0):
            stop = "out_of_budget"
        elif _incomplete(response) == "max_output_tokens":
            stop = "max_tokens"
        if stop is not None:
            # every function call needs an output, or the conversation can't continue
            items.extend(_output(c.call_id, f"Not run: {stop}.", error=True) for c in calls)
            return RunResult(stop, text, items)
        if not calls:
            return RunResult("done" if agent.done else "waiting", text, items)

        for c in calls:
            content, error = invoke(actions, c.name, c.arguments)
            items.append(_output(c.call_id, with_next_step(agent, content), error))
    return RunResult("done" if agent.done else "max_turns", text, items)


def _tool_param(action: Action) -> dict[str, Any]:
    return {
        "type": "function",
        "name": action.name,
        "description": description(action),
        "parameters": arguments(action).model_json_schema(),
        "strict": False,  # strict mode requires every parameter; arguments are validated here
    }


def _incomplete(response: Any) -> str | None:
    if response.status != "incomplete" or response.incomplete_details is None:
        return None
    reason: str | None = response.incomplete_details.reason
    return reason


def _refused(response: Any) -> bool:
    if _incomplete(response) == "content_filter":
        return True
    return any(
        part.type == "refusal"
        for item in response.output
        if item.type == "message"
        for part in item.content
    )


def _output(call_id: str, content: str, error: bool = False) -> dict[str, Any]:
    # Function outputs have no error flag, so errors are marked in the text.
    return {
        "type": "function_call_output",
        "call_id": call_id,
        "output": f"Error: {content}" if error else content,
    }
