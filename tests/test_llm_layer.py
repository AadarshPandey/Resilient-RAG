"""Tests for M4: LLM layer (breaker, client backoff, tier invariants, and provider mapping)."""

from unittest.mock import MagicMock, patch

import pytest

from rag.clock import VirtualClock
from rag.config import Settings
from rag.errors import LLMError, LLMRateLimitError, LLMTimeoutError
from rag.fakes import ScriptedChatModel
from rag.llm.base import LLMResult
from rag.llm.breaker import BreakerState, CircuitBreaker
from rag.llm.client import LLMClient
from rag.llm.providers import build_chat_model


def test_circuit_breaker_state_machine() -> None:
    """Verify circuit breaker transitions: closed -> open -> half_open -> closed / open."""
    clock = VirtualClock(start_time=100.0)
    breaker = CircuitBreaker(
        failure_threshold=3,
        cooldown_seconds=60.0,
        clock=clock,
    )

    assert breaker.state == BreakerState.CLOSED
    assert breaker.can_execute() is True

    # 2 failures do not open the breaker
    breaker.record_failure()
    breaker.record_failure()
    assert breaker.state == BreakerState.CLOSED

    # 3rd failure opens the breaker
    breaker.record_failure()
    assert breaker.state == BreakerState.OPEN
    assert breaker.can_execute() is False

    # Clock moves 30s: still OPEN
    clock.advance(30.0)
    assert breaker.can_execute() is False
    assert breaker.state == BreakerState.OPEN

    # Clock moves another 31s (total 61s > cooldown): transitions to HALF_OPEN
    clock.advance(31.0)
    assert breaker.can_execute() is True
    assert breaker.state == BreakerState.HALF_OPEN

    # Probe succeeds: transitions to CLOSED
    breaker.record_success()
    assert breaker.state == BreakerState.CLOSED
    assert breaker.consecutive_failures == 0

    # Test half-open failure returning to open
    breaker.record_failure()
    breaker.record_failure()
    breaker.record_failure()
    assert breaker.state == BreakerState.OPEN
    clock.advance(65.0)
    assert breaker.can_execute() is True
    assert breaker.state == BreakerState.HALF_OPEN
    breaker.record_failure()
    assert breaker.state == BreakerState.OPEN


def test_client_backoff_schedule_with_virtual_clock() -> None:
    """Verify exponential backoff schedule advances VirtualClock deterministically."""
    clock = VirtualClock(start_time=0.0)
    # Model fails twice with RateLimit, then succeeds
    model = ScriptedChatModel(
        responses=[
            LLMRateLimitError("429 Too Many Requests"),
            LLMRateLimitError("429 Too Many Requests"),
            LLMResult(content="Success after 2 retries", tier=0),
        ]
    )

    client = LLMClient(
        tiers={0: model},
        clock=clock,
        max_retries=3,
        backoff_base=1.0,
    )

    result = client.invoke([{"role": "user", "content": "hi"}], tier=0)
    assert result.content == "Success after 2 retries"
    # Attempt 0 backoff: 1.0 * (2^0) = 1.0s
    # Attempt 1 backoff: 1.0 * (2^1) = 2.0s
    # Total virtual clock advance = 3.0s
    assert clock.now() == 3.0


def test_client_never_switches_tier_on_failure() -> None:
    """Core fairness invariant: LLMClient must NEVER switch tiers on failure."""
    clock = VirtualClock(start_time=0.0)
    tier0_model = ScriptedChatModel(
        responses=[
            LLMRateLimitError("429"),
            LLMRateLimitError("429"),
            LLMRateLimitError("429"),
            LLMRateLimitError("429"),
        ]
    )
    tier1_model = ScriptedChatModel(
        responses=[
            LLMResult(content="Tier 1 answer", tier=1),
        ]
    )

    client = LLMClient(
        tiers={0: tier0_model, 1: tier1_model},
        clock=clock,
        max_retries=2,
        backoff_base=1.0,
    )

    # Calling tier 0 must fail when retries exhaust, NOT silently invoke tier 1
    with pytest.raises(LLMRateLimitError):
        client.invoke([{"role": "user", "content": "query"}], tier=0)

    # Assert tier1 was NEVER called
    assert len(tier1_model.recorded_calls) == 0
    # Tier 0 called initial + 2 retries = 3 calls
    assert len(tier0_model.recorded_calls) == 3


def test_client_non_retryable_error_surfaces_immediately() -> None:
    """Non-retryable errors must fail immediately without backoff delay."""
    clock = VirtualClock(start_time=10.0)
    tier0_model = ScriptedChatModel(
        responses=[
            LLMError("Invalid API Key / Auth failed", retryable=False),
        ]
    )

    client = LLMClient(
        tiers={0: tier0_model},
        clock=clock,
        max_retries=3,
        backoff_base=1.0,
    )

    with pytest.raises(LLMError, match="Auth failed") as exc_info:
        client.invoke([{"role": "user", "content": "test"}], tier=0)

    assert exc_info.value.retryable is False
    # Virtual clock did not advance
    assert clock.now() == 10.0
    assert len(tier0_model.recorded_calls) == 1


def test_provider_adapter_mapping_gemini() -> None:
    """Verify Google / Gemini provider adapter maps API exceptions correctly."""
    settings = Settings(
        _env_file=None,
        gemini_api_key="mock_key",
        llm_chain="google:gemini-2.5-flash",
    )

    with patch("langchain_google_genai.ChatGoogleGenerativeAI") as mock_genai_cls:
        mock_instance = MagicMock()
        mock_genai_cls.return_value = mock_instance

        # Test successful invocation
        mock_ai_message = MagicMock()
        mock_ai_message.content = "Answer text"
        mock_ai_message.response_metadata = {
            "usage_metadata": {"prompt_token_count": 15, "candidates_token_count": 25}
        }
        mock_instance.invoke.return_value = mock_ai_message

        model = build_chat_model("google", "gemini-2.5-flash", settings)
        res = model.invoke([{"role": "user", "content": "hello"}], tier=0)
        assert res.content == "Answer text"
        assert res.prompt_tokens == 15
        assert res.completion_tokens == 25
        assert res.tier == 0

        # Test rate limit exception mapping
        class FakeResourceExhausted(Exception):
            pass

        mock_instance.invoke.side_effect = FakeResourceExhausted("429 Resource has been exhausted")
        with pytest.raises(LLMRateLimitError):
            model.invoke([{"role": "user", "content": "hello"}], tier=0)

        # Test timeout exception mapping
        class FakeDeadlineExceeded(Exception):
            pass

        mock_instance.invoke.side_effect = FakeDeadlineExceeded("Deadline exceeded: 504")
        with pytest.raises(LLMTimeoutError):
            model.invoke([{"role": "user", "content": "hello"}], tier=0)
