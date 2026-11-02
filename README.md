# concordat

**Typed sessions for AI agents.** Agents are objects whose interfaces are protocols,
and the runtime enforces those protocols on the model.

> Status: pre-alpha. Agent classes, the protocol/session/budget kernel, and runners for
> Claude (Anthropic) and OpenAI work. See [docs/design.md](https://github.com/miladsikaroudi/concordat/blob/main/docs/design.md).

## Why

"Never refund before verifying identity" usually lives in a prompt and is hoped for.
Concordat makes it a checked protocol:

- **Protocol fidelity:** the session follows the declared order of turns.
- **Least privilege per step:** the model only gets the tools legal in the current state.
- **Bounded spend:** tokens and tool calls are charged automatically and stay within
  budget, even across handoffs. A dollar budget is charged by your code (`budget.spend(usd=...)`),
  since the runners don't know model prices; a run stops when any limited amount reaches zero.

## Evidence

On [τ-bench](https://github.com/sierra-research/tau2-bench)'s 50 airline tasks, with
simulated customers who push the agent to skip confirmation ("just do it and stop
asking"), counting changes made without the explicit "yes" the airline policy requires:

| Agent model | | Tasks passed | Changes made without a "yes" |
|---|---|---|---|
| Claude Haiku 5.5 (3 runs) | standard agent | 112 / 128 | 50 (in 29 task-runs) |
| | concordat agent | 105 / 128 | **0** |
| gpt-4.1 (1 run) | standard agent | 20 / 50 | 26 (in 13 tasks) |
| | concordat agent | 22 / 50 | **0** |

The standard agent broke the rule with both models; concordat never did. With Haiku,
concordat sometimes transferred a customer who kept refusing to say yes, costing some
task success; after telling it to ask again instead, it matched or beat the standard
agent with gpt-4.1 (not yet rerun with Haiku). With cooperative customers neither agent
broke the rule. Setup, limits and all runs:
[bench/RESULTS.md](https://github.com/miladsikaroudi/concordat/blob/main/bench/RESULTS.md).

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

    @tool(when={"start", "verified", "awaiting_approval"})   # legal in several states
    def order_status(self, order_id: str) -> str:            # no `then`: stays put
        return lookup_order(order_id).status

agent = RefundAgent()          # protocol already checked when the class was defined
agent.allowed_tools()          # {'verify_identity': ...}: all the model gets to see
agent.propose_refund(5000)     # ProtocolViolation, raised before the body runs
```

### See the process

`RefundAgent.diagram()` draws the protocol, straight from the class:

```mermaid
stateDiagram-v2
    direction LR
    state "awaiting_approval" as s0
    state "denied" as s1
    state "refunded" as s2
    state "start" as s3
    state "verified" as s4
    [*] --> s3
    s0 --> s1: approve_refund (human rejected)
    s0 --> s2: approve_refund (human approved)
    s0 --> s0: order_status
    s3 --> s3: order_status
    s3 --> s1: verify_identity
    s3 --> s4: verify_identity
    s4 --> s4: order_status
    s4 --> s0: propose_refund
    s1 --> [*]
    s2 --> [*]
```

### Safe inheritance

Specialise an agent by subclassing it. Adding steps is fine; bypassing the parent's
is rejected when the class is defined:

```python
class FastRefundAgent(RefundAgent):
    @tool(when="start", then="refunded")
    def instant_refund(self, order_id: str) -> str: ...

# ProtocolError: FastRefundAgent breaks RefundAgent's protocol: after instant_refund,
# it reaches 'refunded', which RefundAgent only reaches through
# verify_identity → propose_refund → approve_refund
```

So a team can define a compliant base agent once, and every specialised version
keeps its safety steps.

### Running it with Claude

```bash
pip install "concordat[anthropic]"
```

```python
import anthropic
from concordat.anthropic import run

result = run(RefundAgent(), anthropic.Anthropic(), "Please refund order A-100.")
result.status   # "done", or "waiting" if Claude asked the user something
```

Each turn Claude is offered only the actions legal in the current state (a human
step appears as a request that only the approver can decide), one call per
response. Every response's tokens are charged to the agent's budget, and the next
response is capped to what is left. Illegal or invalid calls go back to Claude as
errors, ending with the actions allowed now, so it can correct itself; after three
rejected calls in a row the run stops as `stuck` instead of looping. See [examples/refund.py](https://github.com/miladsikaroudi/concordat/blob/main/examples/refund.py).

### Running it with OpenAI

The same loop over the Responses API (`pip install "concordat[openai]"`):

```python
import openai
from concordat.openai import run

result = run(RefundAgent(), openai.OpenAI(), "Please refund order A-100.", model="...")
```

### Without classes

The same protocol, using the kernel directly:

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
