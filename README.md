# concordat

**Typed sessions for AI agents.** Agents are objects whose interfaces are protocols,
and the runtime enforces those protocols on the model.

> Status: pre-alpha. Agent classes and the protocol/session/budget kernel work;
> provider adapters (OpenAI, Anthropic) are next. See [docs/design.md](docs/design.md).

## Why

"Never refund before verifying identity" usually lives in a prompt and is hoped for.
Concordat makes it a checked protocol:

- **Protocol fidelity:** the session follows the declared order of turns.
- **Least privilege per step:** the model only gets the tools legal in the current state.
- **Bounded spend:** tokens, tool calls and dollars stay within budget, even across handoffs.

## Example

```python
from concordat import Agent, Budget, human, tool

class RefundAgent(Agent):
    final = {"refunded", "denied"}
    budget = Budget(tool_calls=10)

    @tool(when="start", then={True: "verified", False: "denied"})
    def verify_identity(self, user_id: str) -> bool:
        return lookup(user_id).verified

    @tool(when="verified", then="awaiting_approval")
    def propose_refund(self, amount: float) -> float:
        return amount

    @human(when="awaiting_approval", approve="refunded", reject="denied")
    def approve_refund(self, amount: float) -> str:
        return f"Refund ${amount}?"         # shown to the human approver

agent = RefundAgent()          # protocol already checked when the class was defined
agent.allowed_tools()          # {'verify_identity': ...}: all the model gets to see
agent.propose_refund(5000)     # ProtocolViolation, raised before the body runs
```

The same thing without classes, using the kernel directly:

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
