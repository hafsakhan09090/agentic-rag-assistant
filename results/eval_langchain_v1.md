## Retrieval (embeddings: hashing)

Run on the author's Windows laptop, November 2026:
`python -m agentic_rag eval --agent langchain --provider openai`
Model: openai/gpt-oss-120b via Groq (OpenAI-compatible API), prompt v1, one run.

| Retriever | recall@1 | recall@3 | recall@5 | MRR | mean ms | p95 ms |
|---|---|---|---|---|---|---|
| bm25 | 0.966 | 1.0 | 1.0 | 0.983 | 0.14 | 0.49 |
| vector | 0.931 | 1.0 | 1.0 | 0.96 | 0.36 | 0.61 |
| hybrid | 0.966 | 1.0 | 1.0 | 0.983 | 0.57 | 1.58 |

## Agent (langchain, openai/gpt-oss-120b)

| Metric | Value |
|---|---|
| items | 29 |
| citation hit rate | 1.0 |
| correctness | 0.966 |
| ticket accuracy | 0.0 |
| latency ms mean | 10525.7 |
| latency ms p95 | 19239.4 |
| mean steps | 1.17 |
| fallback steps | 0 |

Agent run used retriever `hybrid` and agent kind `langchain`.
