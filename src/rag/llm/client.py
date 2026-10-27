"""Multi-tier LLM client with per-tier circuit breakers and strict no-tier-switching invariant."""

from typing import Any

from rag.clock import Clock, SystemClock
from rag.errors import LLMError
from rag.llm.base import ChatModel, LLMResult
from rag.llm.breaker import CircuitBreaker


class LLMClient:
    """Multi-tier LLM client.

    CRITICAL INVARIANT:
    This client NEVER switches model tiers on failure. It only retries on the requested tier.
    Tier fallback is strictly an explicit healing action orchestrated by the supervisor.
    """

    def __init__(
        self,
        tiers: dict[int, ChatModel],
        clock: Clock | None = None,
        max_retries: int = 3,
        backoff_base: float = 1.0,
        failure_threshold: int = 3,
        cooldown_seconds: float = 60.0,
    ) -> None:
        self.tiers = tiers
        self.clock = clock or SystemClock()
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.breakers: dict[int, CircuitBreaker] = {
            tier: CircuitBreaker(
                failure_threshold=failure_threshold,
                cooldown_seconds=cooldown_seconds,
                clock=self.clock,
            )
            for tier in tiers
        }

    def invoke(self, messages: list[Any], *, tier: int = 0) -> LLMResult:
        """Invoke LLM on specified tier with retries and circuit breaker protection.

        Args:
            messages: Formatted chat messages or prompts.
            tier: The target model tier (0 = primary, 1 = fallback).

        Returns:
            Standardized LLMResult with content and token counts.

        Raises:
            LLMError: If breaker is open or retries on the tier are exhausted.
        """
        if tier not in self.tiers:
            raise LLMError(f"Tier {tier} is not configured in client", retryable=False)

        breaker = self.breakers[tier]
        if not breaker.can_execute():
            raise LLMError(
                f"Circuit breaker is OPEN for model tier {tier}",
                retryable=False,
            )

        model = self.tiers[tier]
        last_exception: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                result = model.invoke(messages, tier=tier)
                breaker.record_success()
                return result
            except Exception as e:
                last_exception = e
                # Check retryability
                is_retryable = isinstance(e, LLMError) and e.retryable

                if is_retryable and attempt < self.max_retries:
                    # Advance deterministic backoff delay
                    backoff = self.backoff_base * (2**attempt)
                    self.clock.advance(backoff)
                    continue

                # Final failure on this tier
                breaker.record_failure()
                if isinstance(e, LLMError):
                    raise e
                raise LLMError(f"LLM tier {tier} invocation failed: {e}", retryable=False) from e

        if last_exception:
            breaker.record_failure()
            if isinstance(last_exception, LLMError):
                raise last_exception
            raise LLMError(
                f"LLM tier {tier} failed: {last_exception}", retryable=False
            ) from last_exception

        raise LLMError(f"Unexpected termination on tier {tier}", retryable=False)
