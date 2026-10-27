# ADR-001: Self-Healing Supervisor Patterns and Bounded Retry Budget

## Status
Accepted

## Context
Standard RAG architectures suffer from cascade failures: transient rate limits cause hard crashes, hallucinated ungrounded outputs reach users, context overflow silently truncates queries, and poisoned context corrupts answers. Naive recovery patterns rely on unbounded while loops or uncoordinated retries at individual component levels, leading to infinite retry storms, API quota exhaustion, and non-deterministic behavior.

## Decision
We implement a **centralized, stateful Healing Supervisor** governed by strict deterministic rules:
1. **Bounded Budget:** A hard cap of `max_attempts = 4` recovery actions per query.
2. **Deterministic Selector Table:** Every `FailureType` maps to an ordered list of recovery `HealAction` candidates.
3. **No Duplicate Execution:** The supervisor records `(failure, action)` pairs in `heal_history` and never repeats the same action for the same failure.
4. **Terminal Escapes:** When the recovery budget is exhausted or `healing_enabled = False`, the supervisor forces immediate terminal routing (`ABSTAIN` or `ESCALATE`).
5. **Pure State Transitions:** Recovery actions produce pure state patches without modifying global variables.

## Consequences
- **Positive:** Guarantees termination in all failure scenarios; prevents infinite loops and quota drain; ensures reproducible, testable recovery paths.
- **Negative:** Hard budget cap means highly complex cascading outages may terminate in `ABSTAIN` or `ESCALATE` rather than continuing to search for a viable path.
