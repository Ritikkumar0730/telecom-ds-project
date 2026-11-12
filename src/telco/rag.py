"""Retrieval-augmented generation: document chunking, retrieval and retrieval evaluation.

Retrieval uses TF-IDF with sublinear term frequency and bigrams. It runs offline with no model
download. An optional dense backend (sentence-transformers) can be swapped in via `Retriever`.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[2]
POLICY_DIR = ROOT / "docs" / "policies"
EVAL_FILE = ROOT / "docs" / "eval_questions.json"

MAX_CHARS = 700


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    section: str
    text: str


def _split_long(text: str, max_chars: int = MAX_CHARS) -> list[str]:
    """Split on paragraph boundaries so that no chunk exceeds `max_chars` when possible."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    out, buf = [], ""
    for p in paras:
        if buf and len(buf) + len(p) + 2 > max_chars:
            out.append(buf)
            buf = p
        else:
            buf = f"{buf}\n\n{p}" if buf else p
    if buf:
        out.append(buf)
    return out


def load_chunks(policy_dir: Path = POLICY_DIR) -> list[Chunk]:
    """Chunk every Markdown policy file by `##` section, keeping the section title as metadata."""
    chunks: list[Chunk] = []
    for path in sorted(policy_dir.glob("*.md")):
        doc_id = path.stem
        sections = re.split(r"^## ", path.read_text(encoding="utf-8"), flags=re.MULTILINE)
        for raw in sections[1:]:
            title, _, body = raw.partition("\n")
            for piece in _split_long(body.strip()):
                chunks.append(Chunk(doc_id=doc_id, section=title.strip(), text=piece))
    return chunks


class Retriever:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2), stop_words="english", sublinear_tf=True, min_df=1
        )
        # Index title + body so that section names contribute to matching.
        corpus = [f"{c.section}. {c.text}" for c in chunks]
        self.matrix = self.vectorizer.fit_transform(corpus)

    def search(self, query: str, k: int = 3) -> list[tuple[Chunk, float]]:
        q = self.vectorizer.transform([query])
        scores = (self.matrix @ q.T).toarray().ravel()
        order = np.argsort(-scores)[:k]
        return [(self.chunks[i], float(scores[i])) for i in order if scores[i] > 0]


def evaluate_retrieval(retriever: Retriever, qa: list[dict], k: int = 3) -> dict:
    """Recall@k and MRR at document level: a hit means a top-k chunk comes from the expected doc."""
    hits, rr = 0, 0.0
    misses = []
    for item in qa:
        results = retriever.search(item["question"], k=k)
        doc_ranks = [c.doc_id for c, _ in results]
        if item["doc_id"] in doc_ranks:
            hits += 1
            rr += 1.0 / (doc_ranks.index(item["doc_id"]) + 1)
        else:
            misses.append({"question": item["question"], "expected": item["doc_id"], "got": doc_ranks})
    n = len(qa)
    return {
        "n_questions": n,
        f"recall_at_{k}": round(hits / n, 4),
        "mrr": round(rr / n, 4),
        "misses": misses,
    }


def load_eval_questions(path: Path = EVAL_FILE) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


class LSARetriever(Retriever):
    """Latent semantic retrieval: TF-IDF followed by truncated SVD (LSA).

    Maps terms that co-occur across chunks into shared latent dimensions, so paraphrased queries
    can match chunks that share few exact words. Needs no model download. A dense neural
    embedding backend would be the next step up.
    """

    def __init__(self, chunks: list[Chunk], n_components: int = 64):
        from sklearn.decomposition import TruncatedSVD
        from sklearn.preprocessing import normalize

        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True)
        tfidf = self.vectorizer.fit_transform([f"{c.section}. {c.text}" for c in chunks])
        k = min(n_components, tfidf.shape[1] - 1, tfidf.shape[0] - 1)
        self.svd = TruncatedSVD(n_components=k, random_state=0)
        self.matrix = normalize(self.svd.fit_transform(tfidf))
        self._normalize = normalize

    def search(self, query: str, k: int = 3) -> list[tuple[Chunk, float]]:
        q = self._normalize(self.svd.transform(self.vectorizer.transform([query])))
        scores = (self.matrix @ q.T).ravel()
        order = np.argsort(-scores)[:k]
        return [(self.chunks[i], float(scores[i])) for i in order if scores[i] > 0.05]
