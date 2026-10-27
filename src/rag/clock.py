"""Clock abstractions for deterministic time handling and chaos tests."""

import time
from typing import Protocol


class Clock(Protocol):
    """Protocol for time sources."""

    def now(self) -> float:
        """Return the current time in seconds."""
        ...

    def advance(self, seconds: float) -> None:
        """Advance the clock by the given number of seconds."""
        ...


class SystemClock:
    """Real system wall clock using time.time()."""

    def now(self) -> float:
        """Return the actual wall clock time."""
        return time.time()

    def advance(self, seconds: float) -> None:
        """System clock cannot be advanced manually; no-op to avoid sleeps."""
        # Never call time.sleep() in chaos code or tests
        pass


class VirtualClock:
    """Deterministic virtual clock for unit tests and reproducible chaos experiments."""

    def __init__(self, start_time: float = 0.0) -> None:
        self._current_time = float(start_time)

    def now(self) -> float:
        """Return virtual time."""
        return self._current_time

    def advance(self, seconds: float) -> None:
        """Advance virtual time deterministically."""
        if seconds < 0:
            raise ValueError("Cannot move clock backwards in time")
        self._current_time += seconds
