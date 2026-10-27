# ADR-005: RAGAS Evaluation Adapter with Defensive Import Fallback

## Status
Accepted

## Context
RAGAS (Retrieval Augmented Generation Assessment) is the standard library for evaluating faithfulness, answer relevancy, and context metrics. However, `ragas` has heavy dependencies and is known to break or attempt network connections/deprecated imports (e.g. `langchain_community.chat_models.vertexai`) upon module loading. If an evaluation suite crashes because RAGAS or an API key is unavailable, benchmark and CI pipelines fail completely.

## Decision
1. We wrap RAGAS evaluation behind a defensive adapter in `src/rag/evals/ragas_adapter.py`.
2. All imports of `ragas` and its subpackages are performed inside defensive `try/except` blocks.
3. If `ragas` is missing, throws an import error, or fails during metric computation (e.g. missing API key or network failure), the adapter catches the exception, logs a warning, and returns `RagasEvaluationResult(available=False, error=str(e))`.
4. Our primary, zero-cost benchmark metric is the normalized `compute_key_fact_coverage` scorer in `src/rag/evals/golden.py`, which is 100% offline, deterministic, and dependency-free.

## Consequences
- **Positive:** Benchmarks and automated tests never crash due to third-party RAGAS failures; tests run offline without network access; clear visibility when RAGAS metrics are available vs unavailable.
- **Negative:** RAGAS metrics may be skipped if dependencies or keys are misconfigured.
