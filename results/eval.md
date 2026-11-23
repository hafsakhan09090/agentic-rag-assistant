## Retrieval (embeddings: hashing)

| Retriever | recall@1 | recall@3 | recall@5 | MRR | mean ms | p95 ms |
|---|---|---|---|---|---|---|
| bm25 | 0.966 | 1.0 | 1.0 | 0.983 | 0.05 | 0.12 |
| vector | 0.931 | 1.0 | 1.0 | 0.96 | 0.13 | 0.23 |
| hybrid | 0.966 | 1.0 | 1.0 | 0.983 | 0.2 | 0.37 |

## Agent (scripted, rule-based-v1)

| Metric | Value |
|---|---|
| items | 29 |
| citation hit rate | 1.0 |
| correctness | 0.793 |
| ticket accuracy | 1.0 |
| latency ms mean | 0.6 |
| latency ms p95 | 1.9 |
| mean steps | 1.38 |
| fallback steps | 0 |

Agent run used retriever `hybrid` and agent kind `loop`.
