"""Benchmark evaluation runner across golden set questions."""

from pathlib import Path

from pydantic import BaseModel

from rag.bootstrap import Components
from rag.evals.golden import compute_key_fact_coverage, load_golden_set
from rag.evals.ragas_adapter import RagasEvaluationResult, evaluate_with_ragas
from rag.schemas import Outcome


class EvalQuestionResult(BaseModel):
    """Detailed evaluation result for an individual benchmark question."""

    question_id: str
    question: str
    outcome: Outcome
    answer: str
    key_fact_coverage: float
    citations_count: int
    ragas_metrics: RagasEvaluationResult | None = None


class BenchmarkSummary(BaseModel):
    """Aggregate benchmark evaluation summary."""

    total_questions: int
    answered_count: int
    mean_key_fact_coverage: float
    results: list[EvalQuestionResult]


def run_benchmark_eval(
    components: Components,
    golden_path: str | Path = "data/golden/golden_set.jsonl",
    limit: int | None = None,
    use_ragas: bool = False,
) -> BenchmarkSummary:
    """Run pipeline against golden question set and score results.

    Args:
        components: Initialized pipeline components.
        golden_path: Path to golden_set.jsonl.
        limit: Optional maximum questions to evaluate.
        use_ragas: Whether to invoke RAGAS evaluation.

    Returns:
        BenchmarkSummary model.
    """
    questions = load_golden_set(golden_path)
    if limit is not None and limit > 0:
        questions = questions[:limit]

    results: list[EvalQuestionResult] = []
    coverages: list[float] = []

    for q in questions:
        state = components.run(question=q.question, thread_id=f"eval_{q.id}")
        raw_outcome = state.get("outcome") or Outcome.FAILED
        outcome = Outcome(raw_outcome) if isinstance(raw_outcome, str) else raw_outcome
        answer = state.get("answer", "")
        citations = state.get("citations", [])

        coverage = compute_key_fact_coverage(answer, q.key_facts)
        coverages.append(coverage)

        ragas_res: RagasEvaluationResult | None = None
        if use_ragas:
            contexts = [c.text for c in citations if c.text]
            ragas_res = evaluate_with_ragas(
                question=q.question,
                answer=answer,
                contexts=contexts or [answer],
                reference=q.reference_answer,
            )

        results.append(
            EvalQuestionResult(
                question_id=q.id,
                question=q.question,
                outcome=outcome,
                answer=answer,
                key_fact_coverage=coverage,
                citations_count=len(citations),
                ragas_metrics=ragas_res,
            )
        )

    answered_cnt = sum(1 for r in results if r.outcome == Outcome.ANSWERED)
    mean_cov = sum(coverages) / len(coverages) if coverages else 0.0

    return BenchmarkSummary(
        total_questions=len(questions),
        answered_count=answered_cnt,
        mean_key_fact_coverage=mean_cov,
        results=results,
    )
