"""BM25 sparse retriever wrapper."""
from __future__ import annotations

from typing import Any, Dict, List


class BM25Retriever:
    def __init__(self, bm25: Any, meta: List[Dict]):
        self.bm25 = bm25
        self.meta = meta

    def retrieve(self, query: str, topk: int = 10):
        if self.bm25 is None:
            return []
        tokens = query.split()
        scores = list(self.bm25.get_scores(tokens))
        ranked_indices = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)[:topk]
        results = []
        for idx in ranked_indices:
            results.append({"score": float(scores[idx]), "meta": self.meta[idx]})
        return results
