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

When = str | Collection[str]
"""A state name, or the states in which a step is legal."""

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
    """An agent method that is also a protocol action, legal only in the states ``when``."""

    def __init__(self, fn: Callable[Concatenate[Any, P], Any], when: When, targets: Iterable[str]):
        self.fn = fn
        self.when = frozenset({when} if isinstance(when, str) else when)
        if not self.when:
            raise ValueError(f"{fn.__name__}: 'when' names no state")
        self.targets = frozenset(targets)  # empty: the step stays in its current state
        self.name = fn.__name__
        signature = inspect.signature(fn)
        self.signature = signature.replace(parameters=list(signature.parameters.values())[1:])

    def __set_name__(self, owner: type, name: str) -> None:
        self.name = name

    def targets_from(self, state: str) -> frozenset[str]:
        return self.targets or frozenset({state})

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
        self,
        fn: Callable[Concatenate[Any, P], R],
        when: When,
        then: str | Mapping[Any, str] | None,
    ):
        targets = () if then is None else {then} if isinstance(then, str) else then.values()
        super().__init__(fn, when, targets)
        self.then = then

    def _run(self, agent: Agent, *args: P.args, **kwargs: P.kwargs) -> R:
        agent.budget.spend(tool_calls=1)
        result: R = self.fn(agent, *args, **kwargs)
        agent.session.step(self.name, self._target(agent, result))
        return result

    def _target(self, agent: Agent, result: object) -> str:
        if self.then is None:
            return agent.session.state
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
        self, fn: Callable[Concatenate[Any, P], Any], when: When, approve: str, reject: str
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
    when: When, then: str | Mapping[Any, str] | None = None
) -> Callable[[Callable[Concatenate[Any, P], R]], _Tool[P, R]]:
    """A tool the model may call in state ``when`` (a state name, or a set of them).

    ``then`` is the next state, or a mapping from the method's return value to it.
    Leave it out for a step that stays in the current state, such as a lookup.
    """

    def decorate(fn: Callable[Concatenate[Any, P], R]) -> _Tool[P, R]:
        return _Tool(fn, when, then)

    return decorate


def human(
    when: When, approve: str, reject: str
) -> Callable[[Callable[Concatenate[Any, P], Any]], _Human[P]]:
    """A decision a human makes in state ``when`` (a state name, or a set of them).

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
            for state in step.when:
                transitions.setdefault(state, {})[name] = step.targets_from(state)
        try:
            cls.protocol = Protocol(cls.initial, transitions, cls.final)
        except ProtocolError as e:
            raise ProtocolError(f"{cls.__name__}: {e}") from None
        cls._check_substitutable()

    @classmethod
    def _check_substitutable(cls) -> None:
        """A subclass must be safe to use wherever a concrete ancestor is (Liskov)."""
        for base in cls.__mro__[1:]:
            parent: Protocol | None = vars(base).get("protocol")
            if parent is None:
                continue  # an abstract base: no protocol to keep
            problem = cls.protocol.refinement_problem(parent, base.__name__)
            for name, step in vars(base)["_steps"].items():
                if isinstance(step, _Human) and not isinstance(cls._steps.get(name), _Human):
                    problem = problem or f"{name!r} is no longer decided by a human"
            if problem:
                raise ProtocolError(f"{cls.__name__} breaks {base.__name__}'s protocol: {problem}")

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

    @classmethod
    def diagram(cls) -> str:
        """The agent's protocol as a Mermaid state diagram, human decisions labelled."""
        if not cls.final:
            raise TypeError(f"{cls.__name__} is abstract: declare its final states")

        def label(state: str, action: str, target: str) -> str:
            step = cls._steps[action]
            if isinstance(step, _Human):
                outcome = "approved" if target == step.approve else "rejected"
                return f"{action} (human {outcome})"
            return action

        return cls.protocol.to_mermaid(label)

    def allowed_tools(self) -> dict[str, Action]:
        """What the model may do right now; everything else is withheld from it."""
        actions = {}
        for name in sorted(self.session.allowed_actions()):
            step = self._steps[name]
            description = inspect.getdoc(step.fn) or ""
            human = isinstance(step, _Human)
            actions[name] = Action(name, description, getattr(self, name), human)
        return actions
