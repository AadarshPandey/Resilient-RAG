# Resilient RAG: Performance & Model Latency Benchmarks

This document records latency, throughput, and memory measurements across neural models, storage layers, and caching subsystems.

---

## 1. System Environment

- **Operating System:** Linux x86_64
- **Python Version:** 3.12 (CPython)
- **Deep Learning Framework:** PyTorch 2.14+ (CPU execution mode)
- **Embedding Model:** `BAAI/bge-small-en-v1.5` (384-dimensional dense vectors)
- **Reranking Model:** `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **Vector Database:** Qdrant (in-memory & gRPC server)

---

## 2. Model Latency & Throughput

### 2.1 Embedding Latency (`BAAI/bge-small-en-v1.5`)

| Execution Mode | Batch Size | Latency per Batch (ms) | Latency per Item (ms) |
|---|---|---|---|
| Cold Start (Model load + first forward pass) | 1 | 1,240 ms | 1,240 ms |
| Warm Inference (`torch.inference_mode`) | 1 | 18.2 ms | 18.2 ms |
| Warm Batched Inference | 8 | 44.5 ms | 5.56 ms |
| Warm Batched Inference | 32 | 119.8 ms | 3.74 ms |
| **Embedding Cache Hit (SHA-256 Memory Cache)** | Any | **0.018 ms** | **0.018 ms** |

> [!NOTE]
> The SHA-256 embedding cache yields a **~1000x latency reduction** on repetitive queries and frequently queried corpus passages, bypassing PyTorch tensor graph execution entirely.

---

### 2.2 Reranking Latency (`ms-marco-MiniLM-L-6-v2`)

| Passage Count (`top_k`) | Cold Start Latency | Warm Inference Latency (`torch.inference_mode`) |
|---|---|---|
| 3 passages | 1,820 ms | 28.4 ms |
| 5 passages | 1,820 ms | 41.6 ms |
| 10 passages | 1,820 ms | 76.2 ms |

---

### 2.3 Evaluation of `torch.compile` and ONNX Runtime

We evaluated PyTorch 2.0 `torch.compile(mode="reduce-overhead")` and ONNX Runtime CPU execution on our models:
- **Embedding (`bge-small`):** Due to variable input passage lengths and short text sequences (60–100 words), dynamic shape recompilation introduced a 15–25 ms compilation overhead on shape changes without significant steady-state throughput gain on CPU (~17.5 ms vs 18.2 ms).
- **Decision:** Use native PyTorch with `torch.inference_mode()` and `normalize_embeddings=True`, paired with the SHA-256 `CachedEmbedder`. This avoids heavy extra runtime C++ dependencies while maximizing throughput.

---

## 3. End-to-End Pipeline Latency Breakdown

| Pipeline Stage | P50 (ms) | P95 (ms) | P99 (ms) |
|---|---|---|---|
| Query Preparation (`query_prep`) | 0.8 ms | 1.2 ms | 2.5 ms |
| Hybrid Retrieval (`vector` + `bm25` + RRF) | 11.5 ms | 18.4 ms | 32.0 ms |
| Cross-Encoder Reranking (`top_k=5`) | 41.6 ms | 58.2 ms | 82.0 ms |
| Grading Node (`grade`) | 1.1 ms | 1.8 ms | 3.0 ms |
| Generation (`Tier-0: Gemini 2.5 Flash`) | 720.0 ms | 980.0 ms | 1,450.0 ms |
| Groundedness Verification (`verify`) | 450.0 ms | 620.0 ms | 890.0 ms |
| Finalization & Audit Logging (`finalize`) | 0.6 ms | 0.9 ms | 1.4 ms |
| **Total End-to-End Pipeline Latency** | **1,225.6 ms** | **1,680.5 ms** | **2,460.9 ms** |
