from __future__ import annotations

import functools
import inspect
from collections.abc import Callable, Collection, Iterable, Mapping
from dataclasses import dataclass, replace
from typing import Any, ClassVar, Concatenate, Generic, ParamSpec, TypeVar, overload

from .budget import Budget
from .protocol import Protocol, ProtocolError
from .session import ProtocolViolation, Session

P = ParamSpec("P")
R = TypeVar("R")

Approver = Callable[[str, Any], bool]
"""Decides a human step: gets the step name and the method's return value, returns approval."""


def cli_approver(step: str, request: Any) -> bool:
    answer = input(f"[{step}] {request}\nApprove? [y/N] ")
    return answer.strip().lower() in {"y", "yes"}


@dataclass(frozen=True)
class Action:
    """Something the model may do in the current state: call a tool, or ask a human."""

    name: str
    description: str
    call: Callable[..., Any]
    human: bool


class _Step(Generic[P, R]):
    """An agent method that is also a protocol action, legal only in state ``when``."""

    def __init__(self, fn: Callable[Concatenate[Any, P], Any], when: str, targets: Iterable[str]):
        self.fn = fn
        self.when = when
        self.targets = frozenset(targets)
        self.name = fn.__name__
        signature = inspect.signature(fn)
        self.signature = signature.replace(parameters=list(signature.parameters.values())[1:])

    def __set_name__(self, owner: type, name: str) -> None:
        self.name = name

    @overload
    def __get__(self, agent: None, owner: type | None = None) -> _Step[P, R]: ...
    @overload
    def __get__(self, agent: Agent, owner: type | None = None) -> Callable[P, R]: ...
    def __get__(
        self, agent: Agent | None, owner: type | None = None
    ) -> _Step[P, R] | Callable[P, R]:
        if agent is None:
            return self

        @functools.wraps(self.fn)
        def bound(*args: P.args, **kwargs: P.kwargs) -> R:
            agent.session.check(self.name)  # before any side effect
            return self._run(agent, *args, **kwargs)

        bound.__signature__ = self.signature  # type: ignore[attr-defined]  # without self
        return bound

    def _run(self, agent: Agent, *args: P.args, **kwargs: P.kwargs) -> R:
        raise NotImplementedError


class _Tool(_Step[P, R]):
    def __init__(
        self, fn: Callable[Concatenate[Any, P], R], when: str, then: str | Mapping[Any, str]
    ):
        super().__init__(fn, when, {then} if isinstance(then, str) else then.values())
        self.then = then

    def _run(self, agent: Agent, *args: P.args, **kwargs: P.kwargs) -> R:
        agent.budget.spend(tool_calls=1)
        result: R = self.fn(agent, *args, **kwargs)
        agent.session.step(self.name, self._target(result))
        return result

    def _target(self, result: object) -> str:
        if isinstance(self.then, str):
            return self.then
        try:
            return self.then[result]
        except (KeyError, TypeError):
            raise ProtocolViolation(
                f"{self.name!r} returned {result!r}, which leads to no state; "
                f"expected one of {list(self.then)}"
            ) from None


class _Human(_Step[P, bool]):
    def __init__(
        self, fn: Callable[Concatenate[Any, P], Any], when: str, approve: str, reject: str
    ):
        super().__init__(fn, when, {approve, reject})
        self.approve = approve
        self.reject = reject

    def _run(self, agent: Agent, *args: P.args, **kwargs: P.kwargs) -> bool:
        request = self.fn(agent, *args, **kwargs)
        approved = agent.approver(self.name, request)
        agent.session.step(self.name, self.approve if approved else self.reject)
        return approved


def tool(
    when: str, then: str | Mapping[Any, str]
) -> Callable[[Callable[Concatenate[Any, P], R]], _Tool[P, R]]:
    """A tool the model may call in state ``when``.

    ``then`` is the next state, or a mapping from the method's return value to it.
    """

    def decorate(fn: Callable[Concatenate[Any, P], R]) -> _Tool[P, R]:
        return _Tool(fn, when, then)

    return decorate


def human(
    when: str, approve: str, reject: str
) -> Callable[[Callable[Concatenate[Any, P], Any]], _Human[P]]:
    """A decision a human makes in state ``when``.

    The model may request it, but only the approver decides: the method's return value
    is shown to the approver, and calling the method returns the decision.
    """

    def decorate(fn: Callable[Concatenate[Any, P], Any]) -> _Human[P]:
        return _Human(fn, when, approve, reject)

    return decorate


class Agent:
    """Base class whose ``@tool`` and ``@human`` methods define its protocol.

    Declaring ``final`` makes a class concrete: its protocol is built and checked when
    the class is defined. Classes without ``final`` are abstract bases to inherit from.
    """

    instructions: ClassVar[str] = ""  # the system prompt for model-driven runs
    initial: ClassVar[str] = "start"
    final: ClassVar[Collection[str]] = ()
    budget: Budget = Budget()  # a template: each instance gets its own copy
    protocol: ClassVar[Protocol]
    _steps: ClassVar[dict[str, _Step[Any, Any]]] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        steps: dict[str, _Step[Any, Any]] = {}
        for klass in reversed(cls.__mro__):
            for name, attr in vars(klass).items():
                if isinstance(attr, _Step):
                    steps[name] = attr
                else:
                    steps.pop(name, None)  # overridden by a plain method
        cls._steps = steps
        if not cls.final:
            return
        transitions: dict[str, dict[str, frozenset[str]]] = {}
        for name, step in steps.items():
            transitions.setdefault(step.when, {})[name] = step.targets
        try:
            cls.protocol = Protocol(cls.initial, transitions, cls.final)
        except ProtocolError as e:
            raise ProtocolError(f"{cls.__name__}: {e}") from None

    def __init__(self, approver: Approver = cli_approver) -> None:
        if not self.final:
            raise TypeError(f"{type(self).__name__} is abstract: declare its final states")
        self.approver = approver
        self.budget = replace(type(self).budget)
        self.session = Session(self.protocol, self.budget)

    @property
    def state(self) -> str:
        return self.session.state

    @property
    def done(self) -> bool:
        return self.session.done

    def allowed_tools(self) -> dict[str, Action]:
        """What the model may do right now; everything else is withheld from it."""
        actions = {}
        for name in sorted(self.session.allowed_actions()):
            step = self._steps[name]
            description = inspect.getdoc(step.fn) or ""
            human = isinstance(step, _Human)
            actions[name] = Action(name, description, getattr(self, name), human)
        return actions
