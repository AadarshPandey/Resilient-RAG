"""Per-tier circuit breaker with virtual clock support."""

from enum import StrEnum

from rag.clock import Clock, SystemClock


class BreakerState(StrEnum):
    """Operational states of a circuit breaker."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Circuit breaker isolating blast radius per model tier."""

    def __init__(
        self,
        failure_threshold: int = 3,
        cooldown_seconds: float = 60.0,
        clock: Clock | None = None,
    ) -> None:
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self.clock = clock or SystemClock()
        self.state = BreakerState.CLOSED
        self.consecutive_failures: int = 0
        self.last_failure_time: float = 0.0

    def can_execute(self) -> bool:
        """Determine whether an invocation is allowed under current state."""
        if self.state == BreakerState.CLOSED:
            return True

        if self.state == BreakerState.OPEN:
            elapsed = self.clock.now() - self.last_failure_time
            if elapsed >= self.cooldown_seconds:
                self.state = BreakerState.HALF_OPEN
                return True
            return False

        # In HALF_OPEN, allow a single probe
        return True

    def record_success(self) -> None:
        """Record successful invocation, closing breaker and resetting count."""
        self.state = BreakerState.CLOSED
        self.consecutive_failures = 0

    def record_failure(self) -> None:
        """Record failed invocation, updating consecutive count and state."""
        self.consecutive_failures += 1
        self.last_failure_time = self.clock.now()

        if (
            self.state == BreakerState.HALF_OPEN
            or self.consecutive_failures >= self.failure_threshold
        ):
            self.state = BreakerState.OPEN
