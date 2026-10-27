"""Chaos Regression Gate: verify Condition C safe-outcome rate against results/baseline.json."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from rag.bootstrap import build_components
from rag.chaos.loader import load_scenarios
from rag.chaos.runner import ExperimentRunner
from rag.chaos.scorecard import compute_scorecard
from rag.clock import VirtualClock
from rag.evals.golden import load_golden_set
from rag.fakes import FakeVectorStore, HashEmbedder, LexicalReranker
from rag.ingest.chunking import chunk_document
from rag.ingest.loaders import load_file
from rag.retrieval.bm25 import BM25Index
from rag.retrieval.retriever import Retriever
from scripts.generate_baseline_scorecard import GroundedResponderChatModel


def run_chaos_regression(
    baseline_path: str | Path = "results/baseline.json",
    n_questions: int = 5,
    seeds: list[int] | None = None,
) -> bool:
    """Run offline chaos suite with fakes and assert Condition C safe-outcome rate >= baseline."""
    base_file = Path(baseline_path)
    if not base_file.is_file():
        print(f"Error: Baseline file not found at {base_file}", file=sys.stderr)
        return False

    baseline_data: dict[str, Any] = json.loads(base_file.read_text(encoding="utf-8"))
    min_threshold: float = float(baseline_data.get("min_acceptable_safe_outcome_rate", 0.95))

    seeds = seeds or [42, 100]
    questions = load_golden_set("data/golden/golden_set.jsonl")[:n_questions]
    scenarios = load_scenarios("configs/scenarios.yaml")[:5]

    # Index corpus into memory fakes
    embedder = HashEmbedder(dimension=32)
    vstore = FakeVectorStore()
    bm25 = BM25Index()

    corpus_dir = Path("data/raw")
    for f in corpus_dir.glob("*.md"):
        doc = load_file(f)
        chunks = chunk_document(doc, chunk_size=300, chunk_overlap=30)
        vstore.upsert(chunks, embedder.embed([c.text for c in chunks]))
        bm25.index_chunks(chunks)

    clock = VirtualClock(start_time=0.0)
    chat_model = GroundedResponderChatModel()
    retriever = Retriever(
        embedder=embedder,
        vector_store=vstore,
        bm25_index=bm25,
        reranker=LexicalReranker(),
    )
    components = build_components(
        overrides={
            "clock": clock,
            "embedder": embedder,
            "vector_store": vstore,
            "bm25_index": bm25,
            "reranker": LexicalReranker(),
            "retriever": retriever,
            "chat_model": chat_model,
        }
    )

    question_strings = [q.question for q in questions]
    runner = ExperimentRunner()

    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tf:
        temp_runs_file = Path(tf.name)

    print("Running chaos regression gate evaluation for Condition C...")
    try:
        # Run Condition C (Supervisor active)
        runs_c = runner.run_experiment(
            components=components,
            questions=question_strings,
            scenarios=scenarios,
            seeds=seeds,
            conditions=["C"],
            output_file=temp_runs_file,
        )
        scorecard = compute_scorecard(runs_c)
    finally:
        if temp_runs_file.is_file():
            temp_runs_file.unlink()

    # Condition C scorecard entry
    c_entry = scorecard.condition_metrics.get("C")
    if not c_entry:
        print("Error: No Condition C results found in scorecard.", file=sys.stderr)
        return False

    safe_rate = c_entry.safe_outcome_rate
    print(f"Condition C Safe Outcome Rate: {safe_rate:.2%} (Threshold: {min_threshold:.2%})")

    if safe_rate < min_threshold:
        print(
            f"❌ REGRESSION GATE FAILED: Safe outcome rate {safe_rate:.2%} is below baseline {min_threshold:.2%}",
            file=sys.stderr,
        )
        return False

    print("✅ REGRESSION GATE PASSED: Healing paths verified successfully.")
    return True


if __name__ == "__main__":
    success = run_chaos_regression()
    sys.exit(0 if success else 1)
