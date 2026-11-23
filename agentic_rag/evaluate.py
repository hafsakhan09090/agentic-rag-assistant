"""Evaluation.

Retrieval  (all 29 items; log items use the question plus the log's top patterns)
    recall@k  - at least one retrieved chunk belongs to a relevant document
    MRR       - reciprocal rank of the first relevant chunk
    latency   - mean and p95 milliseconds per search

Agent  (same items, full run through the guardrailed agent)
    citation hit  - a cited chunk belongs to a relevant document
    correct       - at least half of the expected keywords appear in the answer
    ticket ok     - for log items, a ticket was created when expected
    latency / steps

Caveat: questions were written by the project author against the same runbooks, so scores are
optimistic. Use them to compare retrievers, models and prompt versions, not as an absolute benchmark.
"""
from __future__ import annotations

import json
import statistics
import tempfile
import time
from pathlib import Path
from typing import Dict, List

from .factory import make_agent
from .tools import analyze_log


def load_questions(path: str) -> List[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _query_for(item: dict) -> str:
    if item.get("log_file"):
        analysis = analyze_log(Path(item["log_file"]).read_text(encoding="utf-8"))
        extra = " ".join(p["pattern"].split(":", 1)[-1] for p in analysis["top_patterns"][:3])
        return f"{item['question']} {extra}"
    return item["question"]


def _p95(values: List[float]) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))] if ordered else 0.0


def evaluate_retrieval(retrievers: Dict[str, object], questions: List[dict], ks=(1, 3, 5)) -> Dict[str, dict]:
    out = {}
    for name, retriever in retrievers.items():
        hits_at = {k: 0 for k in ks}
        rr, times = [], []
        for item in questions:
            query = _query_for(item)
            t0 = time.perf_counter()
            results = retriever.search(query, k=max(ks))
            times.append((time.perf_counter() - t0) * 1000)
            rank = next((h.rank for h in results if h.chunk.doc_id in item["relevant"]), None)
            rr.append(1.0 / rank if rank else 0.0)
            for k in ks:
                if rank and rank <= k:
                    hits_at[k] += 1
        n = len(questions)
        out[name] = {
            **{f"recall@{k}": round(hits_at[k] / n, 3) for k in ks},
            "mrr": round(sum(rr) / n, 3),
            "latency_ms_mean": round(statistics.mean(times), 2),
            "latency_ms_p95": round(_p95(times), 2),
        }
    return out


def evaluate_agent(retriever, questions: List[dict], provider=None, agent_kind: str = "loop") -> dict:
    rows = []
    for item in questions:
        with tempfile.TemporaryDirectory() as tmp:
            agent = make_agent(agent_kind, retriever, str(Path(tmp) / "tickets.jsonl"), provider)
            log = Path(item["log_file"]).read_text(encoding="utf-8") if item.get("log_file") else None
            result = agent.run(item["question"], log)
        answer = result["answer"].lower()
        kws = item["keywords"]
        cited_docs = {c["chunk_id"].split("#")[0] for c in result["citations"]}
        row = {
            "id": item["id"],
            "citation_hit": bool(cited_docs & set(item["relevant"])),
            "correct": sum(1 for k in kws if k.lower() in answer) >= max(1, len(kws) / 2),
            "latency_ms": result["latency_ms"],
            "steps": len(result["steps"]),
            "blocked": result["blocked"],
            "fallback_steps": agent.fallback_steps,
        }
        if "ticket_expected" in item:
            row["ticket_ok"] = (result["tickets_created"] > 0) == item["ticket_expected"]
        rows.append(row)
    n = len(rows)
    ticket_rows = [r for r in rows if "ticket_ok" in r]
    lat = [r["latency_ms"] for r in rows]
    return {
        "planner": agent.name,
        "model": agent.model,
        "items": n,
        "citation_hit_rate": round(sum(r["citation_hit"] for r in rows) / n, 3),
        "correctness": round(sum(r["correct"] for r in rows) / n, 3),
        "ticket_accuracy": round(sum(r["ticket_ok"] for r in ticket_rows) / len(ticket_rows), 3) if ticket_rows else None,
        "latency_ms_mean": round(statistics.mean(lat), 1),
        "latency_ms_p95": round(_p95(lat), 1),
        "mean_steps": round(statistics.mean(r["steps"] for r in rows), 2),
        "fallback_steps": sum(r["fallback_steps"] for r in rows),
        "rows": rows,
    }


def to_markdown(retrieval: Dict[str, dict], agent: dict, note: str = "") -> str:
    lines = ["## Retrieval" + (f" ({note})" if note else ""), "", "| Retriever | recall@1 | recall@3 | recall@5 | MRR | mean ms | p95 ms |", "|---|---|---|---|---|---|---|"]
    for name, m in retrieval.items():
        lines.append(f"| {name} | {m['recall@1']} | {m['recall@3']} | {m['recall@5']} | {m['mrr']} | "
                     f"{m['latency_ms_mean']} | {m['latency_ms_p95']} |")
    lines += ["", f"## Agent ({agent['planner']}, {agent['model']})", "", "| Metric | Value |", "|---|---|"]
    for key in ("items", "citation_hit_rate", "correctness", "ticket_accuracy", "latency_ms_mean",
                "latency_ms_p95", "mean_steps", "fallback_steps"):
        lines.append(f"| {key.replace('_', ' ')} | {agent[key]} |")
    return "\n".join(lines) + "\n"
