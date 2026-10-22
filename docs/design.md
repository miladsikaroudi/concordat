# Concordat — design (v0.1)

*Typed sessions for AI agents: agents are objects whose interfaces are protocols,
and the runtime enforces those protocols on the model.*

## Problem

Agent frameworks let a model call any tool, in any order, at any cost, and check
policy afterwards (if at all). "Never refund before verifying identity" lives in a
prompt and is hoped for. Real deployments need three guarantees the prompt cannot
give:

1. **Protocol fidelity** — the conversation follows a declared order of turns.
2. **Least privilege per step** — the model only sees the tools that are legal *now*.
3. **Bounded spend** — tokens, tool calls and money stay within a declared budget,
   including across handoffs to other agents.

## Idea

Borrow the object-oriented model developers already know, and give it the
discipline of session types / typestate:

| OOP | Concordat |
|---|---|
| class | agent template: protocol + instructions + tools + budget |
| object | a live `Session` with its own state |
| interface | a `Protocol`: states and the actions legal in each |
| method | a tool or turn, callable only in certain states |
| composition | handoff: delegate a sub-session with a slice of the budget |

Unlike an ordinary object, the implementation (the model) can't be trusted to keep
its interface. So the protocol is not documentation — it is enforced on every turn.

## Core concepts

- **Protocol** — a finite state machine: an initial state, final states, and for
  each state the actions allowed and the states each action may lead to. Checked
  when it is defined: no undefined states, no unreachable states, and every state
  can still reach a final state (no dead ends).
- **Action** — a tool call, a model turn, or a human decision (`human.approve`).
  Humans are just another party with actions in the protocol.
- **Session** — a protocol instance. It tracks the current state, exposes
  `allowed_actions()`, and rejects anything else with `ProtocolViolation`.
- **Budget** — tokens, tool calls and USD. Spending is checked; `split()` carves
  out a child budget and removes it from the parent (linear: it can't be spent twice).
- **Handoff** — starting a child session on another agent with a split budget; the
  parent stays in a waiting state until the child finishes.

## Enforcement, per turn

On each model turn the runtime derives from the current state:

1. the **tool list** passed to the provider (only allowed actions), and
2. the **output schema** (structured output / grammar) for that state, and
3. the **remaining budget** (max tokens, stop if exhausted).

Then a **monitor** validates whatever comes back before any side effect runs.
Layer 1–2 prevent most violations; layer 3 catches the rest, so it works even with
providers that offer no decoding control.

## Planned surface (v0.1)

```python
class RefundAgent(Agent):
    budget = Budget(tokens=50_000, tool_calls=10)

    @tool(when="start", then={"verified", "denied"})
    def verify_identity(self, user_id: str) -> bool: ...

    @tool(when="verified", then={"awaiting_approval"})
    def propose_refund(self, amount: float) -> Proposal: ...

    @human(when="awaiting_approval", approve="refunded", reject="denied")
    def approve_refund(self, proposal: Proposal): ...
```

The decorators compile to a `Protocol`; the class is checked when it is defined.

## Scope

**v0.1 (this milestone)**
- `Protocol`, `Session`, `Budget` kernel with definition-time checks — *done in scaffold*
- `Agent` class + decorators on top of the kernel
- Provider adapters: OpenAI, Anthropic (allowed tools + structured output per state)
- Human approval connector (CLI first)
- Showcase: a τ³-bench airline or retail domain, typed vs. untyped agent

**Later**
- Open models via llguidance / XGrammar (true grammar-constrained decoding)
- Multiparty protocols projected to per-agent views
- Durable sessions (journal + resume, possibly on Temporal / Restate / DBOS)
- TypeScript port

**Non-goals**
- Replacing agent frameworks. Concordat should sit inside them.
- Proving model outputs are *correct* — only that they are *allowed*.

## Related work

- **ZipperGen** (Bollig, Függer, Nowak, 2026) — projects message-sequence-chart
  workflows to deadlock-free agent programs; durable replay. Concordat differs by
  constraining the model's turns from the protocol state, budgets, and delegation.
- **XFlow / XPF** (2026) — protocol state gates tools and human approvals at
  runtime. Concordat derives the gates from a checked protocol.
- **Mungo / StMungo** (Gay, Vasconcelos et al.) — typestate for Java objects; the
  closest object-oriented ancestor.
- **Pact** (2026) — choreographic language for agent ecosystems with game theory.
- Runtime guardrails (AgentSpec, ContrAgent, AgentLTL) and durable execution
  (Temporal, Restate, DBOS) solve pieces of the problem separately.

## Open questions

- How to declare "free" regions where the model may loop without strict ordering.
- Whether branching should be chosen by the tool result or by the model.
- How strict to be by default: reject, retry with feedback, or escalate to a human.
