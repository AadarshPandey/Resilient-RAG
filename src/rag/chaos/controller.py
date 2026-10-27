"""Deterministic, seeded chaos controller orchestrating fault schedules."""

import random
from collections import defaultdict

from rag.chaos.faults import FaultConfig


class FaultController:
    """Controls when synthetic faults trigger based on deterministic schedules and seeds."""

    def __init__(
        self,
        faults: list[FaultConfig] | None = None,
        seed: int | None = 42,
    ) -> None:
        self.faults = list(faults or [])
        self.seed = seed
        self._rng = random.Random(seed)
        self._call_counts: dict[str, int] = defaultdict(int)
        self._fired_counts: dict[int, int] = defaultdict(int)

    def reset(self) -> None:
        """Reset internal call and firing counters."""
        self._call_counts.clear()
        self._fired_counts.clear()
        if self.seed is not None:
            self._rng = random.Random(self.seed)

    def check_fault(
        self,
        component: str,
        tier: int | None = None,
        node: str | None = None,
    ) -> FaultConfig | None:
        """Check whether a fault should fire for this invocation.

        Args:
            component: Target component ("llm", "vector", "bm25", "embedder", "retriever").
            tier: Optional model tier.
            node: Optional pipeline node name.

        Returns:
            The FaultConfig to apply, or None.
        """
        key = f"{component}:{tier}:{node}"
        current_call = self._call_counts[key]
        self._call_counts[key] += 1

        for idx, fault in enumerate(self.faults):
            if fault.component != component:
                continue
            if fault.tier is not None and tier is not None and fault.tier != tier:
                continue
            if fault.node is not None and node is not None and fault.node != node:
                continue

            fired = self._fired_counts[idx]
            if current_call >= fault.after_n_calls and (fault.times == -1 or fired < fault.times):
                self._fired_counts[idx] += 1
                return fault

        return None
