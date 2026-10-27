"""Experiment runner for reproducible A/B/C chaos evaluations."""

import json
from pathlib import Path
from typing import Any

from rag.bootstrap import Components
from rag.chaos.controller import FaultController
from rag.chaos.faults import Scenario
from rag.chaos.wrappers import (
    ChaosBM25Index,
    ChaosChatModel,
    ChaosEmbedder,
    ChaosRetriever,
    ChaosVectorStore,
)
from rag.clock import VirtualClock
from rag.schemas import Outcome, RunRecord


class ExperimentRunner:
    """Orchestrates benchmark runs across conditions A, B, and C with incremental persistence."""

    def dry_run(
        self,
        questions: list[str],
        scenarios: list[Scenario],
        seeds: list[int],
        conditions: list[str],
    ) -> dict[str, Any]:
        """Compute execution budget without making any API or model invocations.

        Args:
            questions: List of benchmark question strings.
            scenarios: List of chaos scenarios.
            seeds: List of random seeds.
            conditions: Selected experimental conditions (e.g. ['A', 'B', 'C']).

        Returns:
            Dictionary detailing run counts and estimated LLM call budget.
        """
        total_runs = len(questions) * len(scenarios) * len(seeds) * len(conditions)

        # Baseline: ~2 calls per run (generate + verify). C with healing: ~4 calls per run.
        estimated_calls = 0
        for cond in conditions:
            runs_in_cond = len(questions) * len(scenarios) * len(seeds)
            if cond == "C":
                estimated_calls += runs_in_cond * 4
            else:
                estimated_calls += runs_in_cond * 2

        return {
            "total_runs": total_runs,
            "questions_count": len(questions),
            "scenarios_count": len(scenarios),
            "seeds_count": len(seeds),
            "conditions": conditions,
            "estimated_llm_calls": estimated_calls,
        }

    def load_completed_keys(self, output_path: str | Path) -> set[tuple[str, str, int, str]]:
        """Load already executed (question, scenario, seed, condition) tuples from existing JSONL."""
        path = Path(output_path)
        if not path.is_file():
            return set()

        completed: set[tuple[str, str, int, str]] = set()
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    key = (
                        str(data.get("question")),
                        str(data.get("scenario", "none")),
                        int(data.get("seed", 0)),
                        str(data.get("condition")),
                    )
                    completed.add(key)
                except Exception:
                    continue
        return completed

    def run_experiment(
        self,
        components: Components,
        questions: list[str],
        scenarios: list[Scenario],
        seeds: list[int],
        conditions: list[str],
        output_file: str | Path = "results/runs.jsonl",
    ) -> list[RunRecord]:
        """Execute full experiment across conditions, flushing incrementally to JSONL."""
        out_path = Path(output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        completed = self.load_completed_keys(out_path)

        results: list[RunRecord] = []

        for cond in conditions:
            healing_enabled = cond == "C"

            for scenario in scenarios:
                for seed in seeds:
                    for question in questions:
                        key = (question, scenario.name, seed, cond)
                        if key in completed:
                            continue

                        # Setup clock and fault controller
                        clock = VirtualClock(start_time=0.0)
                        faults = scenario.faults if cond in ("B", "C") else []
                        controller = FaultController(faults=faults, seed=seed)

                        # Wrap components with chaos injection
                        components.embedder = ChaosEmbedder(
                            target=components.embedder,
                            controller=controller,
                            clock=clock,
                        )
                        components.vector_store = ChaosVectorStore(
                            target=components.vector_store,
                            controller=controller,
                            clock=clock,
                        )
                        components.bm25_index = ChaosBM25Index(
                            target=components.bm25_index,
                            controller=controller,
                            clock=clock,
                        )
                        components.retriever = ChaosRetriever(
                            target=components.retriever,
                            controller=controller,
                            clock=clock,
                        )

                        # Wrap active LLM model
                        active_model = components.llm_client.tiers[0]
                        chaos_model = ChaosChatModel(
                            target=active_model,
                            controller=controller,
                            clock=clock,
                        )
                        components.llm_client.tiers[0] = chaos_model
                        components.llm_client.clock = clock

                        # Run query
                        try:
                            state = components.run(
                                question=question,
                                healing_enabled=healing_enabled,
                            )
                            raw_outcome = state.get("outcome") or Outcome.FAILED
                            outcome = (
                                Outcome(raw_outcome)
                                if isinstance(raw_outcome, str)
                                else raw_outcome
                            )
                            answer = state.get("answer", "")
                            citations = state.get("citations", [])
                            heal_actions = [a for _, a in state.get("heal_history", [])]
                            faults_detected = (
                                [str(state.get("last_failure"))]
                                if state.get("last_failure")
                                else []
                            )
                            llm_calls = state.get("llm_calls", 0)
                            tokens = state.get("token_usage", 0)
                            trace = state.get("trace", [])
                        except Exception as e:
                            outcome = Outcome.FAILED
                            answer = f"Error: {e}"
                            citations = []
                            heal_actions = []
                            faults_detected = [type(e).__name__]
                            llm_calls = 1
                            tokens = 0
                            trace = []

                        record = RunRecord(
                            question=question,
                            outcome=outcome,
                            answer=answer,
                            citations=citations,
                            attempts=len(heal_actions) + 1,
                            detected_faults=faults_detected,
                            heal_actions=heal_actions,
                            duration_seconds=clock.now(),
                            trace=trace,
                            tokens_used=tokens,
                            llm_calls=llm_calls,
                            condition=cond,
                            scenario=scenario.name,
                            seed=seed,
                        )

                        # Incremental JSONL flush
                        with open(out_path, "a", encoding="utf-8") as f:
                            f.write(record.model_dump_json() + "\n")

                        completed.add(key)
                        results.append(record)

        return results
