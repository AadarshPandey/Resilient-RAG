# Resilient RAG Chaos Scorecard

| Condition | Runs | Task Success (95% CI) | Safe Outcome (95% CI) | Recovery Rate | MTTR (s) |
|---|---|---|---|---|---|
| A | 140 | 100.0% [97.3%, 100.0%] | 100.0% [97.3%, 100.0%] | 100.0% | 0.00 |
| B | 140 | 14.3% [9.4%, 21.0%] | 92.9% [87.4%, 96.1%] | 91.7% | 0.00 |
| C | 140 | 21.4% [15.4%, 28.9%] | 100.0% [97.3%, 100.0%] | 100.0% | 1.33 |

## Paired Overhead (Condition C vs B)

- **Mean Added Calls:** 0.14
- **Mean Added Tokens:** 9.6
- **Mean Added Latency (s):** 0.00