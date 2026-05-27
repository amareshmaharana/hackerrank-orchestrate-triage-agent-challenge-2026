"""Dense retriever wrapper with graceful fallback when dense dependencies are unavailable."""
from __future__ import annotations

from typing import Dict, List, Sequence
import math

try:  # pragma: no cover - optional dependency
    import numpy as np
except Exception:  # pragma: no cover - fallback when numpy is unavailable
    np = None

try:  # pragma: no cover - optional dependency
    import faiss
except Exception:  # pragma: no cover - fallback when FAISS is unavailable
    faiss = None


class DenseRetriever:
    def __init__(self, index, meta: List[Dict]):
        self.index = index
        self.meta = meta

    def _normalize(self, vector: Sequence[float]):
        norm = math.sqrt(sum(float(value) * float(value) for value in vector)) or 1.0
        return [float(value) / norm for value in vector]

    def retrieve(self, query_emb, topk: int = 10):
        if self.index is None or query_emb is None:
            return []
        if faiss is None or np is None:
            return []
        if getattr(query_emb, "ndim", 1) == 1:
            query_emb = query_emb.reshape(1, -1)
        faiss.normalize_L2(query_emb)
        scores, indices = self.index.search(query_emb.astype("float32"), topk)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.meta):
                continue
            results.append({"score": float(score), "meta": self.meta[idx]})
        return results
