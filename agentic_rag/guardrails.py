"""Guardrails around the agent.

Three layers, none of which depend on the model behaving well:

1. Input checks  - size cap, control-character removal, secret redaction, and a block list of
                   instruction-override attempts in the *user's question*.
2. Untrusted data - log text and retrieved documents are data, never instructions. Lines that look
                   like instructions are replaced before the model or any tool output sees them.
3. Tool policy   - lives in tools.py / agent.py (allowlist, schemas, evidence gate, ticket limit).

Pattern matching is a heuristic and cannot stop every attack. It is one layer of several.
"""
from __future__ import annotations

import re
from typing import List, Tuple

MAX_INPUT_CHARS = 8000
MAX_LOG_CHARS = 20000

_INJECTION = [
    re.compile(r"ignore\s+(all\s+|any\s+|the\s+)?(previous|prior|above|earlier)\s+(instructions|prompts?|rules)", re.I),
    re.compile(r"disregard\s+.{0,40}(instructions|rules|guidelines)", re.I),
    re.compile(r"(reveal|print|show|repeat|output)\s+.{0,40}(system\s+prompt|instructions|api[\s_-]?key|secret|canary)", re.I),
    re.compile(r"\byou\s+are\s+now\b", re.I),
    re.compile(r"</?\s*(system|tool_result|log|assistant)\s*>", re.I),
    re.compile(r"\bnew\s+instructions?\s*:", re.I),
    re.compile(r"(create|open|file)\s+\d{2,}\s+tickets", re.I),
]

_SECRETS = [
    (re.compile(r"\bAIza[0-9A-Za-z_\-]{20,}\b"), "[REDACTED_KEY]"),
    (re.compile(r"\bAQ\.[A-Za-z0-9_\-]{20,}\b"), "[REDACTED_KEY]"),
    (re.compile(r"\bsk-[A-Za-z0-9_\-]{20,}\b"), "[REDACTED_KEY]"),
    (re.compile(r"\bBearer\s+[A-Za-z0-9._\-]{20,}\b"), "Bearer [REDACTED_TOKEN]"),
    (re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"), "[REDACTED_EMAIL]"),
]

_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def looks_like_injection(text: str) -> bool:
    return any(p.search(text) for p in _INJECTION)


def redact(text: str) -> Tuple[str, int]:
    count = 0
    for pattern, repl in _SECRETS:
        text, n = pattern.subn(repl, text)
        count += n
    return text, count


def check_input(question: str) -> Tuple[bool, str, List[str]]:
    """Return (allowed, cleaned_question, events)."""
    events: List[str] = []
    text = _CONTROL.sub("", question or "").strip()
    if not text:
        return False, "", ["empty question"]
    if len(text) > MAX_INPUT_CHARS:
        text = text[:MAX_INPUT_CHARS]
        events.append(f"input truncated to {MAX_INPUT_CHARS} characters")
    text, n = redact(text)
    if n:
        events.append(f"redacted {n} secret(s) from the question")
    if looks_like_injection(text):
        events.append("blocked: instruction-override pattern in the question")
        return False, text, events
    return True, text, events


def sanitize_untrusted(text: str, limit: int = MAX_LOG_CHARS) -> Tuple[str, List[str]]:
    """Clean log text or retrieved content. Instruction-like lines are replaced, not obeyed."""
    events: List[str] = []
    text = _CONTROL.sub("", text or "")
    if len(text) > limit:
        text = text[:limit]
        events.append(f"untrusted text truncated to {limit} characters")
    text, n = redact(text)
    if n:
        events.append(f"redacted {n} secret(s) from untrusted text")
    cleaned, replaced = [], 0
    for line in text.splitlines():
        if looks_like_injection(line):
            cleaned.append("[filtered: instruction-like text]")
            replaced += 1
        else:
            cleaned.append(line)
    if replaced:
        events.append(f"filtered {replaced} instruction-like line(s) from untrusted text")
    return "\n".join(cleaned), events
