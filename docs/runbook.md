# Resilient RAG: Production Runbook & Alerting Guide

This runbook documents operational procedures, alert definitions, and incident response workflows for the Resilient RAG pipeline.

---

## 1. Prometheus Alerting Rules

Deploy these Prometheus alert rules in your Prometheus / VictoriaMetrics alerting configuration (`deploy/prometheus/alerts.yml`):

```yaml
groups:
  - name: resilient_rag_alerts
    rules:
      - alert: LLMCircuitBreakerOpen
        expr: rag_circuit_breaker_state == 2
        for: 5m
        labels:
          severity: critical
          service: resilient-rag
        annotations:
          summary: "LLM Circuit Breaker Open for Tier {{ $labels.tier }}"
          description: "Circuit breaker for LLM Tier {{ $labels.tier }} has been in OPEN state for more than 5 minutes. Downstream requests are triggering fallback."
          runbook_url: "docs/runbook.md#1-llm-circuit-breaker-open"

      - alert: AbstainRateSpike
        expr: sum(rate(rag_requests_total{outcome="abstained"}[5m])) / sum(rate(rag_requests_total[5m])) > 0.25
        for: 5m
        labels:
          severity: warning
          service: resilient-rag
        annotations:
          summary: "RAG Abstain Rate Exceeds 25%"
          description: "Over 25% of requests ended in ABSTAIN outcome over the last 5 minutes. Inspect retrieval quality, poisoning, or context limits."
          runbook_url: "docs/runbook.md#2-abstain-rate-spike"

      - alert: HighRequestLatency
        expr: histogram_quantile(0.95, sum(rate(rag_request_duration_seconds_bucket[5m])) by (le)) > 10.0
        for: 5m
        labels:
          severity: warning
          service: resilient-rag
        annotations:
          summary: "95th Percentile Latency Exceeds 10 Seconds"
          description: "RAG request latency p95 is exceeding 10s. May indicate LLM API throttling, cross-encoder bottleneck, or semaphore queueing."
          runbook_url: "docs/runbook.md#3-high-request-latency"

      - alert: RetrievalBackendDownSpike
        expr: sum(rate(rag_heal_attempts_total{failure="retrieval_backend_down"}[5m])) > 1.0
        for: 3m
        labels:
          severity: critical
          service: resilient-rag
        annotations:
          summary: "Retrieval Backend Failures Detected"
          description: "Supervisor is encountering retrieval backend outages (Qdrant or BM25). Check vector store health."
          runbook_url: "docs/runbook.md#4-retrieval-backend-down"
```

---

## 2. Incident Response Playbooks

### Playbook 1: LLM Circuit Breaker Open (`LLMCircuitBreakerOpen`)

**Symptoms:**
- Metric `rag_circuit_breaker_state{tier="0"} == 2` for over 5 minutes.
- Increased rate of `MODEL_FALLBACK` actions in `rag_heal_attempts_total`.
- Increased requests routing to Tier 1 (`rag_llm_tokens_total{tier="1"}`).

**Triage Steps:**
1. Check provider API status (e.g. Google Gemini / OpenAI outage dashboard).
2. Inspect application JSON logs for `LLMError` or `LLMRateLimitError`:
   ```bash
   docker compose logs api --tail=200 | jq 'select(.level=="ERROR" or .level=="WARNING")'
   ```
3. Verify rate limit quotas on provider console.
4. If Tier 0 is down permanently or severely throttled, reconfigure `RAG_LLM_CHAIN` in environment and restart:
   ```bash
   docker compose up -d --no-deps api
   ```

---

### Playbook 2: Abstain Rate Spike (`AbstainRateSpike`)

**Symptoms:**
- `rag_requests_total{outcome="abstained"}` spikes above 25% of total queries.

**Triage Steps:**
1. Check failure types in `rag_heal_attempts_total`:
   ```promql
   sum by (failure) (rate(rag_heal_attempts_total[10m]))
   ```
2. If `prompt_injection` is high: Inspect LangSmith traces for adversarial prompt injection attempts in incoming requests or corpus data.
3. If `low_relevance` or `insufficient_context` is high:
   - Check if corpus is populated or if recent ingest jobs failed.
   - Run corpus audit: `rag ingest data/raw`.
4. If `ungrounded_answer` is high:
   - LLM generation hallucination rate has spiked. Verify temperature and system prompt instructions.

---

### Playbook 3: High Request Latency (`HighRequestLatency`)

**Symptoms:**
- p95 latency > 10s on `/ask`.

**Triage Steps:**
1. Check active concurrency against `RAG_API_CONCURRENCY_LIMIT`:
   - If requests are queued on the bounded semaphore, consider scaling container replicas.
2. Check LLM provider latency vs local embedding/reranking latency:
   - Examine LangGraph node traces in LangSmith (`retrieve`, `grade`, `generate`, `verify`).
3. If cross-encoder reranking is CPU-bound:
   - Check GPU/CPU utilization with `nvidia-smi` or `top`.
   - Consider reducing `top_k` candidates passed to reranker.

---

### Playbook 4: Retrieval Backend Outage (`RetrievalBackendDownSpike`)

**Symptoms:**
- Multiple `RETRIEVAL_BACKEND_DOWN` failures recorded.

**Triage Steps:**
1. Verify Qdrant container health:
   ```bash
   docker compose ps qdrant
   curl -f http://localhost:6333/readyz
   ```
2. Check Qdrant disk space and collection status:
   ```bash
   curl -s http://localhost:6333/collections
   ```
3. Restart vector store service if unresponsive:
   ```bash
   docker compose restart qdrant
   ```
4. Verify fallback behavior: In hybrid mode, supervisor switches to `BM25_ONLY` when vector backend is unavailable, maintaining safe degraded operation.
