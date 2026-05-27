"""Reranker combining semantic similarity and original retrieval score.

The reranker degrades gracefully when the local environment does not have the
sentence-transformers or numpy packages available.
"""
from __future__ import annotations

import math
from typing import Dict, List

try:  # pragma: no cover - optional dependency
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - fallback when model packages are unavailable
    SentenceTransformer = None

try:  # pragma: no cover - optional dependency
    import numpy as np
except Exception:  # pragma: no cover - fallback when numpy is unavailable
    np = None


class Reranker:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", alpha: float = 0.6):
        self.alpha = alpha
        self.model = SentenceTransformer(model_name) if SentenceTransformer is not None else None

    def _fallback_semantic_score(self, query: str, text: str) -> float:
        query_tokens = {token for token in query.lower().split() if len(token) > 2}
        text_tokens = {token for token in text.lower().split() if len(token) > 2}
        if not query_tokens:
            return 0.0
        overlap = len(query_tokens & text_tokens)
        return overlap / max(len(query_tokens), 1)

    def rerank(self, query: str, candidates: List[Dict], topk: int = 5) -> List[Dict]:
        if not candidates:
            return []

        if self.model is None or np is None:
            merged = []
            for candidate in candidates:
                merged.append(
                    {
                        "score": float(candidate.get("score", 0.0)),
                        "meta": candidate["meta"],
                        "sem_score": self._fallback_semantic_score(query, candidate["meta"].get("text", "")),
                        "orig_score": float(candidate.get("score", 0.0)),
                    }
                )
            return sorted(merged, key=lambda item: item["score"], reverse=True)[:topk]

        texts = [candidate["meta"]["text"] for candidate in candidates]
        emb_q = self.model.encode([query])[0]
        emb_c = self.model.encode(texts)

        def cos(a, b):
            denominator = (np.linalg.norm(a) + 1e-12) * (np.linalg.norm(b) + 1e-12)
            return float(np.dot(a, b) / denominator)

        sem_scores = [cos(emb_q, vector) for vector in emb_c]
        orig = np.array([candidate.get("score", 0.0) for candidate in candidates], dtype=float)
        if orig.max() - orig.min() > 1e-12:
            norm_orig = (orig - orig.min()) / (orig.max() - orig.min())
        else:
            norm_orig = np.zeros_like(orig)

        merged = []
        for index, candidate in enumerate(candidates):
            combined = self.alpha * sem_scores[index] + (1.0 - self.alpha) * float(norm_orig[index])
            merged.append(
                {
                    "score": float(combined),
                    "meta": candidate["meta"],
                    "sem_score": float(sem_scores[index]),
                    "orig_score": float(orig[index]),
                }
            )

        return sorted(merged, key=lambda item: item["score"], reverse=True)[:topk]
