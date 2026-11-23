"""Tools the agent may call, plus the policy that decides whether a call is allowed.

Policy (enforced in code, independent of the model):
  * allowlist     only tools registered in SCHEMAS can run
  * schemas       every argument is type-checked and bounded
  * evidence gate create_ticket is refused until search_runbooks or analyze_log has run
  * ticket limit  at most `max_tickets` tickets per run
"""
from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .guardrails import sanitize_untrusted

SEVERITIES = ("low", "medium", "high", "critical")

# name -> {arg: (type, required, bound)}   bound: max length for str/list, (lo, hi) for int, choices for enum
SCHEMAS: Dict[str, Dict[str, tuple]] = {
    "search_runbooks": {"query": (str, True, 500), "k": (int, False, (1, 5))},
    "analyze_log": {"text": (str, True, 20000)},
    "create_ticket": {
        "title": (str, True, 120),
        "severity": (str, True, SEVERITIES),
        "summary": (str, True, 1000),
        "runbook_ids": (list, False, 5),
    },
}


class ToolError(Exception):
    """A tool call was refused or failed. The message is safe to show to the model."""


def validate_args(name: str, args: Any) -> Dict[str, Any]:
    if name not in SCHEMAS:
        raise ToolError(f"tool '{name}' is not allowed")
    if not isinstance(args, dict):
        raise ToolError("arguments must be an object")
    schema = SCHEMAS[name]
    unknown = set(args) - set(schema)
    if unknown:
        raise ToolError(f"unexpected argument(s): {', '.join(sorted(unknown))}")
    clean: Dict[str, Any] = {}
    for key, (typ, required, bound) in schema.items():
        if key not in args:
            if required:
                raise ToolError(f"missing argument '{key}'")
            continue
        value = args[key]
        if typ is int:
            if isinstance(value, bool) or not isinstance(value, int):
                raise ToolError(f"'{key}' must be an integer")
            if not bound[0] <= value <= bound[1]:
                raise ToolError(f"'{key}' must be between {bound[0]} and {bound[1]}")
        elif typ is str:
            if not isinstance(value, str) or not value.strip():
                raise ToolError(f"'{key}' must be a non-empty string")
            if isinstance(bound, tuple):
                if value not in bound:
                    raise ToolError(f"'{key}' must be one of {', '.join(bound)}")
            elif len(value) > bound:
                raise ToolError(f"'{key}' is too long (max {bound} characters)")
        elif typ is list:
            if not isinstance(value, list) or len(value) > bound or not all(isinstance(v, str) for v in value):
                raise ToolError(f"'{key}' must be a list of at most {bound} strings")
        clean[key] = value
    return clean


# ------------------------------------------------------------- log analysis --

_LINE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2})(?:[.,]\d+)?\s+"
    r"(?P<level>DEBUG|INFO|NOTICE|WARN|WARNING|ERROR|CRITICAL|FATAL)\s+(?P<component>[\w.\-]+):\s*(?P<msg>.*)$"
)
_MASKS = [
    (re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}(?::\d+)?\b"), "<IP>"),
    (re.compile(r"(?i)\b(user\s+)[\w.@-]+"), r"\1<USER>"),
    (re.compile(r"(?<![A-Za-z_])\d+(?:\.\d+)?"), "<NUM>"),
]


def _template(message: str) -> str:
    for pattern, repl in _MASKS:
        message = pattern.sub(repl, message)
    return message


def analyze_log(text: str) -> Dict[str, Any]:
    """Summarise a log excerpt: level counts, time span, most frequent warning/error patterns."""
    levels: Counter = Counter()
    patterns: Counter = Counter()
    sample: Dict[str, str] = {}
    first = last = None
    parsed = 0
    for line in text.splitlines():
        m = _LINE.match(line)
        if not m:
            continue
        parsed += 1
        level = m.group("level").upper()
        level = {"WARNING": "WARN", "CRITICAL": "ERROR", "FATAL": "ERROR", "NOTICE": "INFO"}.get(level, level)
        levels[level] += 1
        ts = m.group("ts")
        first = ts if first is None or ts < first else first
        last = ts if last is None or ts > last else last
        if level in ("WARN", "ERROR"):
            key = f"{level} {m.group('component')}: {_template(m.group('msg'))}"
            patterns[key] += 1
    top = [{"pattern": p, "count": c} for p, c in patterns.most_common(5)]
    return {
        "parsed_lines": parsed,
        "levels": dict(levels),
        "first_timestamp": first,
        "last_timestamp": last,
        "top_patterns": top,
    }


# ----------------------------------------------------------------- toolbox ---

@dataclass
class RunContext:
    evidence_gathered: bool = False
    tickets_created: int = 0
    retrieved_chunk_ids: set = field(default_factory=set)


class ToolBox:
    def __init__(self, retriever, tickets_path: str = "data/tickets.jsonl", max_tickets: int = 1):
        self.retriever = retriever
        self.tickets_path = Path(tickets_path)
        self.max_tickets = max_tickets

    def call(self, name: str, args: Any, ctx: RunContext) -> Dict[str, Any]:
        clean = validate_args(name, args)
        if name == "search_runbooks":
            hits = self.retriever.search(clean["query"], k=clean.get("k", 3))
            ctx.evidence_gathered = True
            results = []
            for h in hits:
                ctx.retrieved_chunk_ids.add(h.chunk.id)
                text, _ = sanitize_untrusted(h.chunk.text, limit=1200)
                results.append({"chunk_id": h.chunk.id, "title": h.chunk.title, "section": h.chunk.heading,
                                "score": round(h.score, 3), "text": text})
            return {"results": results}
        if name == "analyze_log":
            text, _ = sanitize_untrusted(clean["text"])
            ctx.evidence_gathered = True
            return analyze_log(text)
        if name == "create_ticket":
            if not ctx.evidence_gathered:
                raise ToolError("create_ticket refused: gather evidence with search_runbooks or analyze_log first")
            if ctx.tickets_created >= self.max_tickets:
                raise ToolError(f"create_ticket refused: limit of {self.max_tickets} ticket(s) per run reached")
            return self._create_ticket(clean, ctx)
        raise ToolError(f"tool '{name}' is not allowed")

    def _create_ticket(self, args: Dict[str, Any], ctx: RunContext) -> Dict[str, Any]:
        self.tickets_path.parent.mkdir(parents=True, exist_ok=True)
        existing = 0
        if self.tickets_path.exists():
            with self.tickets_path.open("r", encoding="utf-8") as handle:
                existing = sum(1 for _ in handle)
        ticket = {
            "id": f"TCK-{existing + 1:04d}",
            "title": args["title"],
            "severity": args["severity"],
            "summary": args["summary"],
            "runbook_ids": args.get("runbook_ids", []),
            "mock": True,
        }
        with self.tickets_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(ticket) + "\n")
        ctx.tickets_created += 1
        return {"ticket_id": ticket["id"], "status": "created (mock, written to local file)"}
