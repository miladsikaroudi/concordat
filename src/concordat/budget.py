from __future__ import annotations

from dataclasses import dataclass

_KINDS = ("tokens", "tool_calls", "usd")


class BudgetExceeded(RuntimeError):
    """A spend or split would exceed what is left."""


@dataclass
class Budget:
    """What a session may still spend. ``None`` means unlimited."""

    tokens: int | None = None
    tool_calls: int | None = None
    usd: float | None = None

    def spend(self, *, tokens: int = 0, tool_calls: int = 0, usd: float = 0.0) -> None:
        """Deduct the amounts, or raise without deducting anything."""
        amounts = {"tokens": tokens, "tool_calls": tool_calls, "usd": usd}
        for kind, amount in amounts.items():
            left = getattr(self, kind)
            if left is not None and amount > left:
                raise BudgetExceeded(f"{kind}: need {amount}, {left} left")
        for kind, amount in amounts.items():
            left = getattr(self, kind)
            if left is not None:
                setattr(self, kind, left - amount)

    def split(
        self, *, tokens: int | None = None, tool_calls: int | None = None, usd: float | None = None
    ) -> Budget:
        """Carve out a child budget, e.g. for a handoff. The parent loses what the child gets."""
        child = Budget(tokens=tokens, tool_calls=tool_calls, usd=usd)
        for kind in _KINDS:
            if getattr(self, kind) is not None and getattr(child, kind) is None:
                raise BudgetExceeded(f"{kind}: child can't be unlimited when the parent is limited")
        self.spend(tokens=tokens or 0, tool_calls=tool_calls or 0, usd=usd or 0.0)
        return child
