"""Retrieval over a runbook library.

Three retrievers that share one interface (`search(query, k) -> [Hit]`):

* BM25Retriever    keyword ranking (classic lexical baseline)
* VectorRetriever  cosine similarity over vectors from a pluggable embedder
* HybridRetriever  reciprocal rank fusion of the two

Embedders:
* HashingEmbedder          offline, deterministic, lexical (hashed word and bigram counts).
                           It is a vector space model, NOT a semantic neural embedding.
* OpenAICompatEmbedder     calls an /embeddings endpoint for real semantic vectors.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Union

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how", "i", "if", "in", "is", "it",
    "of", "on", "or", "our", "that", "the", "this", "to", "we", "what", "when", "which", "with", "do",
    "does", "my", "me", "should", "can", "so", "than", "then", "there", "they", "you", "your", "was",
}


def stem(token: str) -> str:
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 5 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 4 and token.endswith("ed"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def tokenize(text: str) -> List[str]:
    return [stem(t) for t in _TOKEN.findall(text.lower()) if t not in _STOP]


@dataclass(frozen=True)
class Chunk:
    id: str
    doc_id: str
    title: str
    heading: str
    text: str

    @property
    def index_text(self) -> str:
        return f"{self.title}. {self.heading}. {self.text}"


@dataclass
class Hit:
    chunk: Chunk
    score: float
    rank: int


def load_chunks(docs_dir: Union[str, Path]) -> List[Chunk]:
    """Split every markdown file into chunks, one per '## ' section."""
    chunks: List[Chunk] = []
    for path in sorted(Path(docs_dir).glob("*.md")):
        title = path.stem.replace("-", " ")
        heading, buffer, n = "Overview", [], 0
        lines = path.read_text(encoding="utf-8").splitlines()

        def flush():
            nonlocal n, buffer
            body = " ".join(s.strip() for s in buffer if s.strip())
            if body:
                n += 1
                chunks.append(Chunk(f"{path.stem}#{n}", path.stem, title, heading, body))
            buffer = []

        for line in lines:
            if line.startswith("# "):
                title = line[2:].strip()
            elif line.startswith("## "):
                flush()
                heading = line[3:].strip()
            else:
                buffer.append(line)
        flush()
    return chunks


# ------------------------------------------------------------------ BM25 ----

class BM25Retriever:
    name = "bm25"

    def __init__(self, chunks: Sequence[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = list(chunks)
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(c.index_text)) for c in self.chunks]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avg_len = sum(self.lengths) / max(len(self.lengths), 1)
        df: Counter = Counter()
        for d in self.docs:
            df.update(d.keys())
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def search(self, query: str, k: int = 5) -> List[Hit]:
        terms = tokenize(query)
        scored = []
        for i, doc in enumerate(self.docs):
            score = 0.0
            for t in terms:
                tf = doc.get(t, 0)
                if not tf:
                    continue
                norm = tf + self.k1 * (1 - self.b + self.b * self.lengths[i] / self.avg_len)
                score += self.idf[t] * tf * (self.k1 + 1) / norm
            if score > 0:
                scored.append((score, i))
        scored.sort(key=lambda x: -x[0])
        return [Hit(self.chunks[i], s, r) for r, (s, i) in enumerate(scored[:k], start=1)]


# ------------------------------------------------------------- embedders ----

Vector = Union[Dict[int, float], List[float]]


def cosine(a: Vector, b: Vector) -> float:
    if isinstance(a, dict) and isinstance(b, dict):
        if len(a) > len(b):
            a, b = b, a
        return sum(v * b.get(i, 0.0) for i, v in a.items())
    return sum(x * y for x, y in zip(a, b))  # type: ignore[arg-type]


class HashingEmbedder:
    """Deterministic lexical vectors: hashed unigrams and bigrams, sub-linear tf, L2-normalised."""

    name = "hashing"

    def __init__(self, dim: int = 2048):
        self.dim = dim

    def _features(self, text: str) -> Counter:
        toks = tokenize(text)
        feats = Counter(toks)
        feats.update(f"{a}_{b}" for a, b in zip(toks, toks[1:]))
        return feats

    def embed(self, texts: Sequence[str]) -> List[Vector]:
        out: List[Vector] = []
        for text in texts:
            vec: Dict[int, float] = {}
            for feat, tf in self._features(text).items():
                digest = hashlib.md5(feat.encode("utf-8")).digest()
                idx = int.from_bytes(digest[:4], "little") % self.dim
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                vec[idx] = vec.get(idx, 0.0) + sign * (1.0 + math.log(tf))
            norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
            out.append({i: v / norm for i, v in vec.items()})
        return out


class OpenAICompatEmbedder:
    """Semantic embeddings from any OpenAI-compatible /embeddings endpoint."""

    name = "api-embeddings"

    def __init__(self, api_key=None, base_url=None, model=None, timeout=60):
        self.api_key = api_key or os.environ.get("AGENT_API_KEY") or os.environ.get("LOGLENS_API_KEY", "")
        self.base_url = (base_url or os.environ.get("AGENT_BASE_URL") or os.environ.get("LOGLENS_BASE_URL")
                         or "https://api.openai.com/v1").rstrip("/")
        self.model = model or os.environ.get("AGENT_EMBED_MODEL", "")
        self.timeout = timeout
        if not self.api_key or not self.model:
            raise ValueError("set AGENT_API_KEY (or LOGLENS_API_KEY) and AGENT_EMBED_MODEL to use API embeddings")

    def embed(self, texts: Sequence[str]) -> List[Vector]:
        body = json.dumps({"model": self.model, "input": list(texts)}).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/embeddings",
            data=body,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"embeddings API error {exc.code}: {exc.read().decode('utf-8', 'replace')[:200]}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"could not reach embeddings API: {exc.reason}") from exc
        vectors = [row["embedding"] for row in sorted(data["data"], key=lambda r: r["index"])]
        out = []
        for v in vectors:  # normalise so cosine == dot product
            norm = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / norm for x in v])
        return out


class VectorRetriever:
    def __init__(self, chunks: Sequence[Chunk], embedder=None):
        self.chunks = list(chunks)
        self.embedder = embedder or HashingEmbedder()
        self.name = f"vector:{self.embedder.name}"
        self.vectors = self.embedder.embed([c.index_text for c in self.chunks])

    def search(self, query: str, k: int = 5) -> List[Hit]:
        qv = self.embedder.embed([query])[0]
        scored = sorted(((cosine(qv, v), i) for i, v in enumerate(self.vectors)), key=lambda x: -x[0])
        return [Hit(self.chunks[i], s, r) for r, (s, i) in enumerate(scored[:k], start=1) if s > 0]


class HybridRetriever:
    """Reciprocal rank fusion: score = sum over retrievers of 1 / (60 + rank)."""

    name = "hybrid"

    def __init__(self, retrievers: Sequence, rrf_k: int = 60):
        self.retrievers = list(retrievers)
        self.rrf_k = rrf_k

    def search(self, query: str, k: int = 5) -> List[Hit]:
        fused: Dict[str, float] = {}
        chunk_by_id: Dict[str, Chunk] = {}
        for retriever in self.retrievers:
            for hit in retriever.search(query, k=max(k * 3, 10)):
                fused[hit.chunk.id] = fused.get(hit.chunk.id, 0.0) + 1.0 / (self.rrf_k + hit.rank)
                chunk_by_id[hit.chunk.id] = hit.chunk
        ranked = sorted(fused.items(), key=lambda x: -x[1])[:k]
        return [Hit(chunk_by_id[cid], score, r) for r, (cid, score) in enumerate(ranked, start=1)]


class SentenceTransformerEmbedder:
    """Semantic embeddings from a Hugging Face sentence-transformers model, run locally.

    Needs: pip install sentence-transformers. The first run downloads the model (about 90 MB for
    the default). Override the model with AGENT_HF_MODEL.
    """

    name = "hf"

    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("sentence-transformers is not installed: pip install sentence-transformers") from exc
        self.model_name = os.environ.get("AGENT_HF_MODEL", model)
        self.model = SentenceTransformer(self.model_name)

    def embed(self, texts: Sequence[str]) -> List[Vector]:
        rows = self.model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)
        return [[float(x) for x in row] for row in rows]


def to_dense(vec: Vector, dim: int) -> List[float]:
    """ChromaDB needs dense vectors; convert the sparse hashing vectors when necessary."""
    if isinstance(vec, dict):
        dense = [0.0] * dim
        for i, v in vec.items():
            dense[i] = v
        return dense
    return list(vec)


class ChromaRetriever:
    """Stores chunk embeddings in a ChromaDB collection and queries it.

    Needs: pip install chromadb. Embeddings are computed by our own embedder and passed in
    explicitly, so Chroma never downloads a model of its own. The collection is rebuilt on start-up
    (66 chunks take well under a second). Pass `path` to keep the database on disk.
    """

    def __init__(self, chunks: Sequence[Chunk], embedder=None, path=None, collection: str = "runbooks"):
        try:
            import chromadb
        except ImportError as exc:
            raise RuntimeError("chromadb is not installed: pip install chromadb") from exc
        self.chunks = list(chunks)
        self.by_id = {c.id: c for c in self.chunks}
        self.embedder = embedder or HashingEmbedder()
        self.name = f"chroma:{self.embedder.name}"
        self.dim = getattr(self.embedder, "dim", None)

        client = chromadb.PersistentClient(path=str(path)) if path else chromadb.EphemeralClient()
        name = f"{collection}_{self.embedder.name}".replace(":", "_")
        try:
            client.delete_collection(name)
        except Exception:  # the collection may not exist yet
            pass
        try:
            self.collection = client.create_collection(name=name, metadata={"hnsw:space": "cosine"})
            self.space = "cosine"
        except Exception:
            try:
                client.delete_collection(name)
            except Exception:
                pass
            self.collection = client.create_collection(name=name)
            self.space = "l2"  # Chroma's default; for unit vectors, squared L2 = 2 - 2*cosine

        vectors = self.embedder.embed([c.index_text for c in self.chunks])
        dense = [to_dense(v, self.dim or 0) for v in vectors]
        self.collection.upsert(
            ids=[c.id for c in self.chunks],
            documents=[c.text for c in self.chunks],
            embeddings=dense,
            metadatas=[{"doc_id": c.doc_id, "title": c.title, "heading": c.heading} for c in self.chunks],
        )

    def search(self, query: str, k: int = 5) -> List[Hit]:
        qv = to_dense(self.embedder.embed([query])[0], self.dim or 0)
        res = self.collection.query(query_embeddings=[qv], n_results=min(k, len(self.chunks)))
        hits = []
        for rank, (cid, dist) in enumerate(zip(res["ids"][0], res["distances"][0]), start=1):
            score = 1.0 - dist if self.space == "cosine" else 1.0 - dist / 2.0
            hits.append(Hit(self.by_id[cid], score, rank))
        return hits


def build_retrievers(docs_dir: Union[str, Path], embedder=None, use_chroma: bool = False, chroma_path=None) -> Dict[str, object]:
    chunks = load_chunks(docs_dir)
    bm25 = BM25Retriever(chunks)
    vector = VectorRetriever(chunks, embedder)
    retrievers: Dict[str, object] = {"bm25": bm25, "vector": vector, "hybrid": HybridRetriever([bm25, vector])}
    if use_chroma:
        retrievers["chroma"] = ChromaRetriever(chunks, embedder, path=chroma_path)
    return retrievers
