# ADR-004: Virtual Chaos Injection with Deterministic Clock

## Status
Accepted

## Context
Validating self-healing behavior requires injecting complex faults: transient rate limits, timeout delays, poisoned chunks, backend outages, and corrupted JSON. Running chaos experiments against live networks or using `time.sleep` causes:
- Slow, flaky, non-deterministic test suites.
- High API token bills.
- Irreproducible test failures across CI environments.

## Decision
1. We implement **Virtual Chaos Injection** via component wrappers (`ChaosEmbedder`, `ChaosVectorStore`, `ChaosBM25Index`, `ChaosChatModel`, `ChaosRetriever`).
2. Timing is simulated using a `VirtualClock` protocol. When a delay fault occurs, the clock advances its virtual timestamp instantaneously without executing wall-clock sleep (`time.sleep` is forbidden in automated tests).
3. Fault scheduling is governed by `FaultController` initialized with a fixed random seed. For a given `(scenario, seed, question)`, the exact sequence of faults is 100% reproducible.
4. Experiments support three conditions:
   - **Condition A (Ceiling):** No faults injected.
   - **Condition B (Baseline):** Faults injected with self-healing disabled.
   - **Condition C (Resilient RAG):** Faults injected with self-healing active.

## Consequences
- **Positive:** Sub-second test execution; 100% reproducible fault schedules; zero network calls or API costs during automated evaluation.
- **Negative:** Virtual clock requires all time-dependent components (breakers, clients, traces) to accept an injected clock protocol rather than calling `time.time()`.
