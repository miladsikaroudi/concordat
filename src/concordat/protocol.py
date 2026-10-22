from __future__ import annotations

from collections.abc import Iterable, Mapping


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
