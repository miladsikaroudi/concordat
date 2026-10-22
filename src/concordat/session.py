from __future__ import annotations

from .budget import Budget
from .protocol import Protocol


class ProtocolViolation(RuntimeError):
    """An action that the protocol does not allow in the current state."""


class Session:
    """A live run of a protocol: tracks the state and rejects illegal actions."""

    def __init__(self, protocol: Protocol, budget: Budget | None = None) -> None:
        self.protocol = protocol
        self.budget = budget if budget is not None else Budget()
        self.state = protocol.initial
        self.history: list[tuple[str, str, str]] = []

    @property
    def done(self) -> bool:
        return self.state in self.protocol.final

    def allowed_actions(self) -> frozenset[str]:
        return self.protocol.actions(self.state)

    def step(self, action: str, to: str | None = None) -> None:
        """Take ``action`` and move to ``to`` (optional when the action has one target)."""
        if action not in self.allowed_actions():
            raise ProtocolViolation(
                f"{action!r} is not allowed in state {self.state!r}; "
                f"allowed: {sorted(self.allowed_actions())}"
            )
        targets = self.protocol.targets(self.state, action)
        if to is None and len(targets) == 1:
            (to,) = targets
        if to is None or to not in targets:
            raise ProtocolViolation(
                f"{action!r} in {self.state!r} can lead to {sorted(targets)}, not {to!r}"
            )
        self.history.append((self.state, action, to))
        self.state = to
