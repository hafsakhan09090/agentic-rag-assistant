"""Command line interface: ask, eval, redteam, index."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import evaluate as ev
from . import redteam as rt
from .factory import make_agent
from .providers import ProviderError, get_provider
from .retrieval import OpenAICompatEmbedder, SentenceTransformerEmbedder, build_retrievers, load_chunks

DOCS = "docs/runbooks"
QUESTIONS = "data/eval_questions.json"


def _embedder(name: str):
    if name == "hf":
        return SentenceTransformerEmbedder()
    if name == "api":
        return OpenAICompatEmbedder()
    return None  # offline hashing vectors


def _retrievers(args):
    use_chroma = args.retriever == "chroma" or args.chroma
    retrievers = build_retrievers(DOCS, _embedder(args.embeddings), use_chroma=use_chroma, chroma_path=args.chroma_path)
    return retrievers[args.retriever], retrievers


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="agentic_rag", description="Agentic RAG assistant with evaluation and guardrails")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument("--provider", default="scripted", choices=["scripted", "openai", "anthropic"])
        p.add_argument("--model", default=None)
        p.add_argument("--agent", default="loop", choices=["loop", "langchain"],
                       help="'loop' is the hand-written agent, 'langchain' runs the same tools through LangChain")
        p.add_argument("--retriever", default="hybrid", choices=["bm25", "vector", "hybrid", "chroma"])
        p.add_argument("--embeddings", default="hashing", choices=["hashing", "hf", "api"],
                       help="hashing: offline lexical vectors; hf: Hugging Face sentence-transformers; api: embeddings endpoint")
        p.add_argument("--chroma", action="store_true", help="also build the ChromaDB retriever (used by eval)")
        p.add_argument("--chroma-path", default=None, help="directory for a persistent ChromaDB (default: in memory)")

    a = sub.add_parser("ask", help="ask a question, optionally with a log file")
    a.add_argument("question")
    a.add_argument("--log", default=None, help="path to a log excerpt")
    a.add_argument("--tickets", default="data/tickets.jsonl")
    a.add_argument("--json", action="store_true", help="print the full result as JSON")
    common(a)

    e = sub.add_parser("eval", help="evaluate retrieval and the agent")
    e.add_argument("--out", default="results/eval.md")
    common(e)

    r = sub.add_parser("redteam", help="run the guardrail red-team suite")
    r.add_argument("--out", default="results/redteam.md")
    common(r)

    sub.add_parser("index", help="show how the runbooks are chunked")

    k = sub.add_parser("lccheck", help="test LangChain against your model, step by step")
    k.add_argument("--provider", default="openai", choices=["openai"])
    k.add_argument("--model", default=None)

    args = parser.parse_args(argv)
    try:
        if args.command == "index":
            chunks = load_chunks(DOCS)
            print(f"{len({c.doc_id for c in chunks})} runbooks, {len(chunks)} chunks")
            for c in chunks[:8]:
                print(f"  {c.id:40s} {c.heading}")
            return 0

        if args.command == "lccheck":
            from .lc_agent import diagnose

            for line in diagnose(get_provider(args.provider, args.model)):
                print(line)
            return 0

        if args.agent == "langchain" and args.provider == "scripted":
            raise ValueError("--agent langchain needs a model: add --provider openai (or anthropic)")

        retriever, retrievers = _retrievers(args)
        provider = get_provider(args.provider, args.model)

        if args.command == "ask":
            log = Path(args.log).read_text(encoding="utf-8") if args.log else None
            agent = make_agent(args.agent, retriever, args.tickets, provider)
            result = agent.run(args.question, log)
            if args.json:
                print(json.dumps(result, indent=2))
            else:
                print(result["answer"], "\n")
                print("Citations:", ", ".join(c["chunk_id"] for c in result["citations"]) or "none",
                      f"(confidence {result['confidence']})")
                if result["tickets_created"]:
                    print(f"Tickets created: {result['tickets_created']} (see {args.tickets})")
                for event in result["guardrail_events"]:
                    print("Guardrail:", event)
                print(f"{result['planner']}, {len(result['steps'])} step(s), {result['latency_ms']} ms")
            return 0

        if args.command == "eval":
            questions = ev.load_questions(QUESTIONS)
            retrieval = ev.evaluate_retrieval(retrievers, questions)
            agent_report = ev.evaluate_agent(retriever, questions, provider, args.agent)
            text = ev.to_markdown(retrieval, agent_report, f"embeddings: {args.embeddings}")
            text += f"\nAgent run used retriever `{args.retriever}` and agent kind `{args.agent}`.\n"
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(text, encoding="utf-8")
            Path(args.out).with_suffix(".json").write_text(
                json.dumps({"retrieval": retrieval, "agent": agent_report}, indent=2), encoding="utf-8")
            print(text)
            return 0

        if args.command == "redteam":
            report = rt.run_redteam(retriever, provider, args.agent)
            text = rt.to_markdown(report, f"{args.agent}/{args.provider}")
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(text, encoding="utf-8")
            print(text)
            return 0 if report["passed"] == report["total"] else 2
    except (ProviderError, ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
