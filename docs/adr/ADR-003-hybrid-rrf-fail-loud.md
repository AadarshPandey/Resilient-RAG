# ADR-003: Hybrid RRF Retrieval and Fail-Loud Semantics

## Status
Accepted

## Context
Production RAG systems frequently combine lexical search (BM25) with dense vector embeddings (e.g. Qdrant) via Reciprocal Rank Fusion (RRF). When one backend fails (e.g. Qdrant vector backend times out or BM25 index throws an I/O error), some systems silently drop the failed backend and return single-backend results without alerting the pipeline.
This silent degradation anti-pattern is dangerous:
- The system continues operating in an degraded state with lower recall without operator awareness.
- Search quality subtly deteriorates without triggering alerts.

## Decision
1. In `HYBRID` retrieval mode, `Retriever` queries both vector and BM25 backends.
2. If **either** backend encounters an error, the `Retriever` immediately **fails loud** by raising `BackendError`. It does not silently fall back to single-backend search.
3. Fallback to `BM25_ONLY` or `VECTOR_ONLY` is an explicit, supervised recovery decision made by the Healing Supervisor (`apply_bm25_only`, `apply_vector_only`), which sets `retrieval_mode` in the pipeline state and logs a trace event.
4. RRF ranking is strictly deterministic using `k=60` and resolves ties by chunk ID.

## Consequences
- **Positive:** No hidden failures; full audit trail of retrieval backend health; degraded retrieval modes are explicit and measurable in scorecard metrics.
- **Negative:** Requires supervisor invocation to handle single-backend outages.
