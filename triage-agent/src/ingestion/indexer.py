"""Indexing: persisted vector index plus BM25 corpus and metadata.

The implementation keeps a dense path when FAISS/numpy are available and
falls back to a pure-Python BM25 scorer when third-party packages are missing.
"""
from __future__ import annotations

import json
import math
import os
from collections import Counter, defaultdict
from typing import Dict, List, Sequence

try:  # pragma: no cover - optional dependency
    import numpy as np
except Exception:  # pragma: no cover - fallback when numpy is unavailable
    np = None

try:  # pragma: no cover - optional dependency
    import faiss
except Exception:  # pragma: no cover - fallback when FAISS is unavailable
    faiss = None


class _FallbackBM25Okapi:
    def __init__(self, corpus: Sequence[Sequence[str]], k1: float = 1.5, b: float = 0.75):
        self.corpus = [list(doc) for doc in corpus]
        self.k1 = k1
        self.b = b
        self.doc_len = [len(doc) for doc in self.corpus]
        self.avgdl = sum(self.doc_len) / max(len(self.doc_len), 1)
        self.doc_freqs = []
        self.idf = {}
        self._build()

    def _build(self):
        document_frequencies = defaultdict(int)
        for document in self.corpus:
            frequencies = Counter(document)
            self.doc_freqs.append(frequencies)
            for token in frequencies:
                document_frequencies[token] += 1
        total_docs = len(self.corpus)
        for token, freq in document_frequencies.items():
            self.idf[token] = math.log(1 + (total_docs - freq + 0.5) / (freq + 0.5))

    def get_scores(self, query_tokens: Sequence[str]):
        scores = []
        query_terms = Counter(query_tokens)
        for doc_index, doc_freq in enumerate(self.doc_freqs):
            score = 0.0
            doc_len = self.doc_len[doc_index] or 1
            for token, qf in query_terms.items():
                if token not in doc_freq:
                    continue
                idf = self.idf.get(token, 0.0)
                freq = doc_freq[token]
                numerator = freq * (self.k1 + 1.0)
                denominator = freq + self.k1 * (1.0 - self.b + self.b * doc_len / max(self.avgdl, 1e-12))
                score += idf * (numerator / denominator)
            scores.append(score)
        return scores


class Indexer:
    def __init__(self, index_path: str = "data/index"):
        self.index_path = os.path.abspath(index_path)
        os.makedirs(self.index_path, exist_ok=True)
        self.faiss_index_path = os.path.join(self.index_path, "faiss.index")
        self.meta_path = os.path.join(self.index_path, "meta.jsonl")
        self.bm25_path = os.path.join(self.index_path, "bm25.json")

    def _to_matrix(self, embeddings: Sequence[Sequence[float]]):
        if np is None:
            return [list(map(float, row)) for row in embeddings]
        return np.asarray(embeddings, dtype="float32")

    def build_index(self, chunks: List[Dict], embeddings):
        with open(self.meta_path, "w", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

        tokenized = [chunk["text"].split() for chunk in chunks]
        with open(self.bm25_path, "w", encoding="utf-8") as f:
            json.dump({"corpus": [" ".join(tokens) for tokens in tokenized]}, f)

        if faiss is None or np is None or embeddings is None:
            return
        try:
            if len(embeddings) == 0:
                return
        except TypeError:
            return
        matrix = self._to_matrix(embeddings)
        if getattr(matrix, "size", 0) == 0:
            return
        if getattr(matrix, "ndim", 1) == 1:
            matrix = matrix.reshape(1, -1)
        faiss.normalize_L2(matrix)
        index = faiss.IndexFlatIP(matrix.shape[1])
        index.add(matrix)
        faiss.write_index(index, self.faiss_index_path)

    def load(self):
        meta = []
        with open(self.meta_path, "r", encoding="utf-8") as f:
            for line in f:
                meta.append(json.loads(line))
        with open(self.bm25_path, "r", encoding="utf-8") as f:
            bm25_json = json.load(f)
        bm25_corpus = [row.split() for row in bm25_json["corpus"]]
        bm25 = _FallbackBM25Okapi(bm25_corpus) if _FallbackBM25Okapi is not None else None

        index = None
        if faiss is not None and np is not None and os.path.exists(self.faiss_index_path):
            try:
                index = faiss.read_index(self.faiss_index_path)
            except Exception:
                index = None
        return index, meta, bm25
