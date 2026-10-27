# ADR-002: Per-Tier Circuit Breakers with No Automated Tier Switching in Client

## Status
Accepted

## Context
When multi-tier LLM configurations (e.g. tier-0: Gemini 1.5 Flash, tier-1: Gemini 1.5 Pro) encounter errors (e.g. HTTP 429 rate limits or 503 outages), a common anti-pattern is for the low-level LLM client to silently switch to tier-1 on its own.
This violates separation of concerns:
1. It hides outages from observability and the supervisory graph.
2. It causes unexpected cloud cost spikes (expensive tier-1 models silently handling high-volume traffic).
3. It makes client behavior non-deterministic and difficult to test.

## Decision
1. We implement **per-tier Circuit Breakers** (`CLOSED`, `OPEN`, `HALF_OPEN`) with cooldowns tracked via an injected `Clock`.
2. The low-level `LLMClient` retries only transient errors on the *same tier* using exponential backoff.
3. The `LLMClient` **strictly never changes `active_tier` autonomously**. If tier-0 fails or trips its breaker, it raises `LLMError` or `LLMRateLimitError` immediately.
4. Switching tiers is exclusively the responsibility of the **Healing Supervisor** via the explicit `MODEL_FALLBACK` recovery action.

## Consequences
- **Positive:** Full audit visibility of when and why tier failovers happen; predictable cost control; clean architectural separation between execution and policy.
- **Negative:** Requires stateful graph coordination to invoke tier fallbacks rather than a self-contained client fallback.
