# concordat

**Typed sessions for AI agents.** Agents are objects whose interfaces are protocols,
and the runtime enforces those protocols on the model.

> Status: pre-alpha. The protocol/session/budget kernel works; the `Agent` class and
> provider adapters are next. See [docs/design.md](docs/design.md).

## Why

"Never refund before verifying identity" usually lives in a prompt and is hoped for.
Concordat makes it a checked protocol:

- **Protocol fidelity:** the session follows the declared order of turns.
- **Least privilege per step:** the model only gets the tools legal in the current state.
- **Bounded spend:** tokens, tool calls and dollars stay within budget, even across handoffs.

## Example

```python
from concordat import Budget, Protocol, Session

refund = Protocol(
    "start",
    {
        "start": {"verify_identity": {"verified", "denied"}},
        "verified": {"propose_refund": {"awaiting_approval"}},
        "awaiting_approval": {"human.approve": {"refunded"}, "human.reject": {"denied"}},
    },
    final={"refunded", "denied"},
)  # checked here: no dead ends, no unreachable states

session = Session(refund, Budget(tokens=50_000, tool_calls=10))
session.allowed_actions()           # {'verify_identity'}: only these tools go to the model
session.step("propose_refund")      # ProtocolViolation: not allowed in state 'start'
```

Handing work to another agent splits the budget, and the parent can't spend it twice:

```python
child_budget = session.budget.split(tokens=10_000, tool_calls=3)
```

## Development

```bash
uv sync
uv run pytest
uv run mypy src
uv run ruff check
```

## License

Apache-2.0
