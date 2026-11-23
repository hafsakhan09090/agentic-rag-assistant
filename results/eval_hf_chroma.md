## Retrieval (embeddings: hf)

Run on the author's Windows laptop (CPU), November 2026: `python -m agentic_rag eval --embeddings hf --chroma`
Model: sentence-transformers/all-MiniLM-L6-v2

| Retriever | recall@1 | recall@3 | recall@5 | MRR | mean ms | p95 ms |
|---|---|---|---|---|---|---|
| bm25 | 0.966 | 1.0 | 1.0 | 0.983 | 0.17 | 0.49 |
| vector | 0.966 | 1.0 | 1.0 | 0.983 | 20.37 | 28.56 |
| hybrid | 0.966 | 1.0 | 1.0 | 0.983 | 21.23 | 31.62 |
| chroma | 0.966 | 1.0 | 1.0 | 0.983 | 19.11 | 32.03 |

## Agent (scripted, rule-based-v1)

| Metric | Value |
|---|---|
| items | 29 |
| citation hit rate | 1.0 |
| correctness | 0.897 |
| ticket accuracy | 1.0 |
| latency ms mean | 23.0 |
| latency ms p95 | 34.7 |
| mean steps | 1.38 |
| fallback steps | 0 |

Agent run used retriever `hybrid` and agent kind `loop`.
