"""Typed sessions for AI agents."""

from importlib.metadata import version as _version

from .agent import Action, Agent, Approver, cli_approver, human, tool
from .budget import Budget, BudgetExceeded
from .protocol import Protocol, ProtocolError
from .session import ProtocolViolation, Session

__all__ = [
    "Action",
    "Agent",
    "Approver",
    "Budget",
    "BudgetExceeded",
    "Protocol",
    "ProtocolError",
    "ProtocolViolation",
    "Session",
    "cli_approver",
    "human",
    "tool",
]
__version__ = _version("concordat")  # set in pyproject.toml
