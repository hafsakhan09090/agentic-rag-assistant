# Agentic RAG Assistant with Evaluation & Guardrails

**Python | LangChain | Hugging Face | ChromaDB (vector database) | function calling**

A troubleshooting assistant for engineers. Give it a question and, optionally, a log excerpt. It searches a library of runbooks, summarises the log, can open a (mock) ticket, and answers with citations to the runbook sections it used.

It is built around three ideas: retrieval you can measure, tool use you can constrain, and guardrails that do not depend on the model behaving well. The core uses only the Python standard library. LangChain, Hugging Face sentence-transformers and ChromaDB are optional integrations.

## What is in it

| Part | What it does |
|---|---|
| Runbook library | 14 short runbooks (database pool exhaustion, disk full, TLS expiry, brute force and others), split into 66 section-level chunks |
| Retrieval | BM25 keyword search, vector search, hybrid (reciprocal rank fusion), and an optional ChromaDB retriever |
| Agent | A bounded tool-calling loop (rule-based planner offline, or an LLM planner), plus an optional LangChain version that uses the same tools and guardrails |
| Tools | `search_runbooks`, `analyze_log`, `create_ticket` (writes to a local file, nothing is sent anywhere) |
| Guardrails | Input checks, secret redaction, untrusted-data filtering, tool allowlist, argument schemas, evidence gate, ticket limit, step limit, citation validation |
| Evaluation | recall@k, MRR, latency, citation hit rate, answer correctness and ticket accuracy on 29 labelled items |
| Red-team suite | 18 attack cases against the guardrails |

## Quick start

Python 3.10 or newer, no packages to install.

```bash
python -m agentic_rag index                                   # show how runbooks are chunked
python -m agentic_rag ask "Requests are queuing and I see 'wait time exceeded'. What do I check?"
python -m agentic_rag ask "What is going on? Escalate if needed." --log data/logs/tls_expired.log
python -m agentic_rag eval                                    # writes results/eval.md
python -m agentic_rag redteam                                 # writes results/redteam.md
python -m unittest discover -s tests -v                         # 48 tests
```

`ask` prints the answer, the cited chunk ids, any guardrail events and the number of steps. Add `--json` for the full result. Tickets are appended to `data/tickets.jsonl`.

## Using a model

By default the agent uses an offline rule-based planner. To let an LLM choose the tool calls:

```bash
export AGENT_API_KEY=your_key
export AGENT_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai   # any OpenAI-compatible API
export AGENT_MODEL=your_model_name
python -m agentic_rag ask "..." --provider openai
python -m agentic_rag eval --provider openai
python -m agentic_rag redteam --provider openai
```

On Windows cmd use `set NAME=value`, one line at a time. The names `LOGLENS_API_KEY`, `LOGLENS_BASE_URL` and `LOGLENS_MODEL` are accepted as fallbacks. For Anthropic set `ANTHROPIC_API_KEY` and use `--provider anthropic`. Keep keys in environment variables and never commit them.

For semantic embeddings instead of the offline vectors, set `AGENT_EMBED_MODEL` and add `--embeddings api` to any command. This path has not been run against a live service yet.

## How it works

```
question (+ optional log)
   -> input checks: size cap, secret redaction, instruction-override block list
   -> loop (max 6 steps):
        planner proposes JSON action  ->  policy check  ->  tool runs  ->  result returned as untrusted data
   -> final answer: citations are checked against chunks actually retrieved in this run
```

### Retrieval

- **Chunking.** Each runbook is split by `##` sections. A chunk id looks like `disk-full#3`.
- **BM25.** Standard keyword ranking with light stemming and stop-word removal.
- **Vector search.** The default embedder hashes words and word pairs into a fixed-size vector and compares them with cosine similarity. It is deterministic and offline, but it is lexical, not semantic. Hugging Face sentence-transformers or an embeddings API can replace it (see Optional integrations).
- **Hybrid.** Reciprocal rank fusion of the two rankings.

### Agent and tools

The planner returns one JSON action per step, either a tool call or a final answer. Tool results are passed back wrapped as untrusted data. The offline planner is extractive: it assembles the answer from retrieved runbook text. The LLM planner asks the model for each action, retries once on invalid output, then lets the offline planner take that step.

### Guardrails

None of these rely on the model cooperating:

| Layer | Rule |
|---|---|
| Input | Truncate to 8,000 characters, strip control characters, redact API keys, bearer tokens and emails, block instruction-override patterns in the question |
| Untrusted data | Log text and retrieved content are sanitised. Lines that look like instructions are replaced with a marker |
| Allowlist | Only the three registered tools can run |
| Schemas | Every argument is type-checked and bounded. Unknown arguments are rejected |
| Evidence gate | `create_ticket` is refused until a search or log analysis has run |
| Ticket limit | One ticket per run |
| Step limit | The loop stops after 6 steps |
| Citations | Citations to chunks that were not retrieved in this run are dropped, and confidence falls to low |
| Output | An internal marker planted in the system prompt is removed if a model ever echoes it |

Pattern matching is a heuristic. It stops obvious attacks and is one layer of several, not a guarantee.

## Optional integrations

Install only what you want to try. Without them everything above still works.

```bash
pip install -r requirements-optional.txt
```

### Hugging Face embeddings (semantic vectors)

```bash
python -m agentic_rag eval --embeddings hf
```

Runs `sentence-transformers/all-MiniLM-L6-v2` locally (set `AGENT_HF_MODEL` to use another). The first run downloads the model. This replaces the lexical hashing vectors, so the `vector` and `hybrid` rows of the evaluation change.

### ChromaDB retriever

```bash
python -m agentic_rag eval --chroma                       # adds a 'chroma' row to the retrieval table
python -m agentic_rag ask "..." --retriever chroma
python -m agentic_rag eval --chroma --embeddings hf --chroma-path .chroma
```

Chunks are embedded with the selected embedder and stored in a ChromaDB collection with their text and metadata. The collection is rebuilt on start-up. Embeddings are passed in explicitly, so Chroma does not download a model of its own.

### LangChain agent

```bash
python -m agentic_rag ask "What is going on? Escalate if needed." --log data/logs/disk_full.log --agent langchain --provider openai
python -m agentic_rag eval --agent langchain --provider openai
python -m agentic_rag redteam --agent langchain --provider openai
```

LangChain runs the tool-calling loop. The same `ToolBox` underneath enforces the allowlist, schemas, evidence gate and ticket limit, the question and log are sanitised first, and citations are parsed from the answer and kept only if they were retrieved in this run. The log excerpt is read through an `analyze_log` tool instead of being pasted into the prompt. It needs a model, so there is no offline mode.

### What is tested

Unit tests use small stand-in modules for each library to check our glue code (vector conversion, scoring, policy enforcement, citation handling, install hints). The real libraries have also been run once each on the author's machine: Hugging Face sentence-transformers and ChromaDB (see Evaluation), and LangChain with `openai/gpt-oss-120b` on Groq. Use `python -m agentic_rag lccheck` to test LangChain against your own model before running the agent.

## Evaluation

Data: 29 labelled items, 24 questions and 5 log-troubleshooting cases, each mapped to the runbook that should be found, plus keywords expected in a correct answer. Raw outputs are in `results/`.

### Retrieval

Run on the author's Windows laptop (CPU) with `python -m agentic_rag eval --embeddings hf --chroma`. The Hugging Face model is `sentence-transformers/all-MiniLM-L6-v2`. Full output: `results/eval_hf_chroma.md`.

| Retriever | recall@1 | recall@3 | recall@5 | MRR | mean ms per query |
|---|---|---|---|---|---|
| BM25 | 0.966 | 1.0 | 1.0 | 0.983 | 0.17 |
| Vector, offline hashing vectors | 0.931 | 1.0 | 1.0 | 0.960 | 0.36 |
| Vector, Hugging Face MiniLM | 0.966 | 1.0 | 1.0 | 0.983 | 20.4 |
| Hybrid (BM25 + MiniLM) | 0.966 | 1.0 | 1.0 | 0.983 | 21.2 |
| ChromaDB (MiniLM embeddings) | 0.966 | 1.0 | 1.0 | 0.983 | 19.1 |

All methods are within one question out of 29 of each other. The labelled set is small and its wording overlaps the runbooks, so it cannot show whether semantic retrieval beats keyword retrieval. What it does show is cost: neural embeddings take about 100 times longer per query than BM25 here. A harder set of paraphrased questions is needed to see a difference. The hashing row comes from `results/eval.md`, which uses the offline vectors.

### Agent, hand-written loop with the offline rule-based planner

With MiniLM retrieval (`results/eval_hf_chroma.md`):

| Metric | Value |
|---|---|
| Citation hit rate | 1.0 |
| Answer correctness (at least half of expected keywords) | 0.897 |
| Ticket accuracy on log cases | 1.0 |
| Mean steps per run | 1.38 |

With the default offline vectors the correctness score is 0.793 (`results/eval.md`). The planner is extractive and its ticket rule was adjusted after the first run, so these scores say little about generation quality.

### Agent, LangChain with an LLM

`python -m agentic_rag eval --agent langchain --provider openai`, model `openai/gpt-oss-120b` through Groq's OpenAI-compatible API, prompt v1, one run. Retrieval used the offline hashing vectors. Full output: `results/eval_langchain_v1.md`.

| Metric | Value |
|---|---|
| Citation hit rate | 1.0 |
| Answer correctness | 0.966 |
| Ticket accuracy on log cases | **0.0** |
| Mean steps per run | 1.17 |
| Mean latency | 10.5 s (hosted model, network included) |

The model never opened a ticket on the five log cases. The evaluation question for those cases ("What is going on and what should I do?") does not ask for escalation, and the v1 prompt told the model to open a ticket only when asked or when the incident is "high or critical". When a question did say "escalate if needed", the agent opened a ticket. The most likely cause is a vague rule in the prompt. Prompt v2 states the rule more precisely (20 or more warning or error lines, or a service failing). It is selectable with `AGENT_PROMPT_VERSION=v2`. It was written after seeing the v1 result on this same set, so any improvement would not be independent evidence. v2 has not been evaluated yet.

Correctness cannot be compared with the rule-based planner: that planner copies runbook text, while the model writes its own answer.

### Red-team

18 of 18 cases pass with the offline planner (`results/redteam.md`). The cases cover instruction overrides, tag spoofing, role override, secrets in questions and logs, instructions hidden in logs, oversized input and arguments, unknown tools, ticket flooding, forged citations, marker leakage and runaway loops. Most cases use a scripted malicious planner, so they test the enforcement code and not the behaviour of any particular model. `redteam --provider openai` sends the input and log-injection cases through a real model; those results are not included here.

### Findings from real runs

- A model wrote runbook ids with a Unicode non-breaking hyphen (`disk‑full#1`). The first citation parser read this as `full#1`, found no match, and the guardrail correctly dropped the citation and lowered confidence. The parser now normalises Unicode dashes.
- Gemini 3.x models rejected the LangChain tool-call round trip with "Function call is missing a thought_signature", because LangChain's OpenAI client does not send that field back. `python -m agentic_rag lccheck` tests plain call, tool request and tool-result round trip separately, which isolated the problem. The hand-written loop is not affected because it does not use native tool-calling messages.

### How to read these numbers

- The questions were written by the author against the same runbooks, and the library is small. High recall is expected. This is a way to compare retrievers and changes, not a benchmark.
- Retrieval is judged at document level. A hit means a chunk from the right runbook was returned.
- One model, one run, one prompt version. LLM output is not deterministic.

## Project layout

```
agentic_rag/
  retrieval.py    chunking, BM25, vector search, hybrid
  agent.py        planners and the guarded loop
  lc_agent.py     the same agent run through LangChain
  factory.py      builds either agent
  tools.py        tool schemas, policy, log analysis, mock tickets
  guardrails.py   input checks, redaction, untrusted-data filtering
  evaluate.py     retrieval and agent metrics
  redteam.py      attack cases
  providers.py    LLM providers (standard library HTTP)
  cli.py          ask, eval, redteam, index
docs/runbooks/    the runbook library
data/             labelled questions and log excerpts
results/          latest evaluation output
tests/            48 tests
```

## Limitations

- The runbook library is small and was written for this project.
- The default vectors are lexical. Paraphrases with no shared words will be missed.
- Ticket creation is a local mock. There is no integration with a ticketing system.
- Evaluation is document-level, with keyword matching for correctness. There is no human or model grading of answers.
- Guardrail patterns are English-only heuristics.
- The LLM planner of the hand-written loop and the API embeddings path have been tested only with fake providers. LangChain was run against one model (`gpt-oss-120b`) and one prompt version; Gemini 3.x is not compatible with LangChain's OpenAI client (see Findings).

## Next steps

- Evaluate prompt v2, and the hand-written loop with an LLM planner, on the same labelled set.
- Run `redteam --agent langchain --provider openai` and record which cases a real model fails.
- Add more runbooks and a held-out question set written by someone else.
- Grade answer quality with a rubric.
- Connect `analyze_log` to a real log pipeline.
