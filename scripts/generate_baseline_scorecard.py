"""Generate reproducible offline baseline scorecard data across conditions A, B, and C."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rag.bootstrap import build_components
from rag.chaos.faults import Scenario
from rag.chaos.loader import load_scenarios
from rag.chaos.runner import ExperimentRunner
from rag.chaos.scorecard import compute_scorecard, scorecard_to_csv, scorecard_to_markdown
from rag.clock import VirtualClock
from rag.evals.golden import load_golden_set
from rag.fakes import FakeVectorStore, HashEmbedder, LexicalReranker
from rag.ingest.chunking import chunk_document
from rag.ingest.loaders import load_file
from rag.llm.base import LLMResult
from rag.retrieval.bm25 import BM25Index
from rag.retrieval.retriever import Retriever
from scripts.plot_results import plot_scorecard


class GroundedResponderChatModel:
    """Mock ChatModel that outputs grounded JSON responses matching questions."""

    def __init__(self) -> None:
        self.call_count: int = 0

    def invoke(self, messages: list[Any], *, tier: int = 0) -> LLMResult:
        import re

        self.call_count += 1
        prompt = " ".join(str(m) for m in messages)

        # Query rewrite step
        if "rewrite the user's question" in prompt.lower():
            return LLMResult(
                content="kubernetes cluster node architecture",
                prompt_tokens=30,
                completion_tokens=10,
                tier=tier,
            )

        # Verification step
        if "claims to verify" in prompt.lower() or "is_grounded" in prompt.lower():
            if "malicious" in prompt.lower() or "poisoned" in prompt.lower():
                verify_json = '{"is_grounded": false, "unsupported_claims": ["unsupported fact"], "summary": "Poisoned chunk rejected"}'
            else:
                verify_json = '{"is_grounded": true, "unsupported_claims": [], "summary": "All claims grounded in context"}'
            return LLMResult(
                content=verify_json,
                prompt_tokens=40,
                completion_tokens=20,
                tier=tier,
            )

        # Generation step
        chunk_ids = re.findall(r"\[([a-zA-Z0-9_\-]+)\]:", prompt)
        citation_id = chunk_ids[0] if chunk_ids else "chunk_k8s_default"

        gen_json = (
            f'{{"answer": "Kubernetes components coordinate containerized workloads reliably across worker nodes.", '
            f'"claims": ["Kubernetes components coordinate containerized workloads reliably"], '
            f'"citations": ["{citation_id}"]}}'
        )
        return LLMResult(
            content=gen_json,
            prompt_tokens=50,
            completion_tokens=25,
            tier=tier,
        )


def generate_baseline_data() -> None:
    clock = VirtualClock(start_time=0.0)
    embedder = HashEmbedder(dimension=32)
    vstore = FakeVectorStore()
    bm25 = BM25Index()
    reranker = LexicalReranker()

    # Index all 36 Kubernetes reference documents from data/raw
    raw_dir = Path("data/raw")
    files = sorted(raw_dir.glob("*.md"))
    all_chunks = []
    for f in files:
        doc = load_file(f)
        chunks = chunk_document(doc, chunk_size=300, chunk_overlap=30)
        all_chunks.extend(chunks)

    vstore.upsert(all_chunks, embedder.embed([c.text for c in all_chunks]))
    bm25.index_chunks(all_chunks)

    retriever = Retriever(
        embedder=embedder,
        vector_store=vstore,
        bm25_index=bm25,
        reranker=reranker,
    )

    chat_model = GroundedResponderChatModel()

    components = build_components(
        overrides={
            "clock": clock,
            "embedder": embedder,
            "vector_store": vstore,
            "bm25_index": bm25,
            "reranker": reranker,
            "retriever": retriever,
            "chat_model": chat_model,
        }
    )

    # Load 10 golden questions
    golden_questions = load_golden_set("data/golden/golden_set.jsonl")[:10]
    question_strings = [q.question for q in golden_questions]

    # Load scenarios
    scenarios: list[Scenario] = load_scenarios("configs/scenarios.yaml")

    runner = ExperimentRunner()
    output_file = Path("results/runs.jsonl")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if output_file.is_file():
        output_file.unlink()

    print(
        f"Running baseline chaos experiments on {len(question_strings)} questions across 14 scenarios..."
    )
    records = runner.run_experiment(
        components=components,
        questions=question_strings,
        scenarios=scenarios,
        seeds=[42],
        conditions=["A", "B", "C"],
        output_file=output_file,
    )

    print(f"Total runs executed: {len(records)}")

    # Scorecard
    scorecard = compute_scorecard(records)
    md_str = scorecard_to_markdown(scorecard)
    csv_str = scorecard_to_csv(scorecard)

    md_path = Path("results/scorecard.md")
    csv_path = Path("results/scorecard.csv")
    png_path = Path("results/scorecard.png")

    md_path.write_text(md_str, encoding="utf-8")
    csv_path.write_text(csv_str, encoding="utf-8")
    print(f"Scorecard saved to {md_path} and {csv_path}")

    # Plot
    plot_scorecard(csv_path, png_path)
    print(f"Scorecard plot saved to {png_path}")


if __name__ == "__main__":
    generate_baseline_data()
