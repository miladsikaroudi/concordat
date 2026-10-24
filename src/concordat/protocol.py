from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable, Mapping


def _where(path: list[str]) -> str:
    return f"after {' → '.join(path)}," if path else "at the start,"


class ProtocolError(ValueError):
    """The protocol definition is inconsistent."""


class Protocol:
    """States, the actions legal in each state, and where each action may lead.

    Checked when constructed: every state must be reachable from ``initial`` and
    must still be able to reach a final state, so a session can never get stuck.
    """

    def __init__(
        self,
        initial: str,
        transitions: Mapping[str, Mapping[str, Iterable[str]]],
        final: Iterable[str],
    ) -> None:
        self.initial = initial
        self.final = frozenset(final)
        self.transitions: dict[str, dict[str, frozenset[str]]] = {
            state: {action: frozenset(targets) for action, targets in actions.items()}
            for state, actions in transitions.items()
        }
        self._check()

    @property
    def states(self) -> frozenset[str]:
        targets = (
            t for actions in self.transitions.values() for ts in actions.values() for t in ts
        )
        return frozenset({self.initial, *self.final, *self.transitions, *targets})

    def actions(self, state: str) -> frozenset[str]:
        return frozenset(self.transitions.get(state, {}))

    def targets(self, state: str, action: str) -> frozenset[str]:
        return self.transitions.get(state, {}).get(action, frozenset())

    def to_mermaid(self, label: Callable[[str, str, str], str] | None = None) -> str:
        """A Mermaid state diagram of the protocol; GitHub renders it in Markdown.

        ``label(state, action, target)`` names each arrow; the action by default.
        """
        ids = {state: f"s{i}" for i, state in enumerate(sorted(self.states))}
        lines = ["stateDiagram-v2", "    direction LR"]
        lines += [f'    state "{state}" as {ids[state]}' for state in sorted(self.states)]
        lines.append(f"    [*] --> {ids[self.initial]}")
        for state, actions in sorted(self.transitions.items()):
            for action, targets in sorted(actions.items()):
                for target in sorted(targets):
                    text = label(state, action, target) if label else action
                    lines.append(f"    {ids[state]} --> {ids[target]}: {text}")
        lines += [f"    {ids[state]} --> [*]" for state in sorted(self.final)]
        return "\n".join(lines)

    def route(self, starts: Iterable[str], goal: str) -> list[str] | None:
        """The shortest sequence of actions from any of ``starts`` to ``goal``."""
        paths: dict[str, list[str]] = {state: [] for state in sorted(starts)}
        queue = deque(paths)
        while queue:
            state = queue.popleft()
            if state == goal:
                return paths[state]
            for action, targets in sorted(self.transitions.get(state, {}).items()):
                for target in sorted(targets - paths.keys()):
                    paths[target] = [*paths[state], action]
                    queue.append(target)
        return None

    def refinement_problem(self, parent: Protocol, name: str = "the parent") -> str | None:
        """Why this protocol can't safely stand in for ``parent``, or None if it can.

        It can if it never takes the parent's actions out of order, never reaches one of
        the parent's states by a route the parent doesn't allow, and only ends where the
        parent can end. Extra actions and states of its own are fine.
        """
        if extra := sorted(self.final - parent.final):
            return f"it can end in {extra[0]!r}, which is not one of {name}'s outcomes"
        parent_actions = {a for actions in parent.transitions.values() for a in actions}

        # Walk this protocol while tracking which parent states each run corresponds to.
        def enter(
            state: str, possible: frozenset[str], path: list[str]
        ) -> tuple[frozenset[str], str | None]:
            if state not in parent.states:
                return possible, None  # a state of its own, between the parent's states
            if state in possible:
                return frozenset({state}), None
            route = parent.route(possible, state)
            how = f"only reaches through {' → '.join(route)}" if route else "can't reach there"
            return possible, f"{_where(path)} it reaches {state!r}, which {name} {how}"

        start, problem = enter(self.initial, frozenset({parent.initial}), [])
        if problem:
            return problem
        queue = deque([(self.initial, start, list[str]())])
        seen = {(self.initial, start)}
        while queue:
            state, possible, path = queue.popleft()
            for action, targets in sorted(self.transitions.get(state, {}).items()):
                after = possible
                if action in parent_actions:
                    after = frozenset(t for p in possible for t in parent.targets(p, action))
                    if not after:
                        return (
                            f"{_where(path)} it allows {action!r}, which {name} doesn't allow there"
                        )
                for target in sorted(targets):
                    narrowed, problem = enter(target, after, [*path, action])
                    if problem:
                        return problem
                    if (target, narrowed) not in seen:
                        seen.add((target, narrowed))
                        queue.append((target, narrowed, [*path, action]))
        return None

    def _check(self) -> None:
        problems = []
        if not self.final:
            problems.append("no final states")
        for state in sorted(self.final):
            if self.transitions.get(state):
                problems.append(f"final state {state!r} has actions")
        for state, actions in self.transitions.items():
            for action, targets in actions.items():
                if not targets:
                    problems.append(f"action {action!r} in {state!r} leads nowhere")

        reachable = {self.initial}
        frontier = [self.initial]
        while frontier:
            for targets in self.transitions.get(frontier.pop(), {}).values():
                for target in targets - reachable:
                    reachable.add(target)
                    frontier.append(target)
        if unreachable := self.states - reachable:
            problems.append(f"unreachable states: {sorted(unreachable)}")

        can_finish = set(self.final)
        changed = True
        while changed:
            changed = False
            for state, actions in self.transitions.items():
                if state not in can_finish and any(ts & can_finish for ts in actions.values()):
                    can_finish.add(state)
                    changed = True
        if stuck := self.states - can_finish:
            problems.append(f"states that can never finish: {sorted(stuck)}")

        if problems:
            raise ProtocolError("; ".join(problems))
