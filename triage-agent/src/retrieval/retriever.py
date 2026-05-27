"""Unified retriever that loads indices and performs hybrid retrieval + reranking."""
from __future__ import annotations

from typing import Dict, List, Optional

from src.ingestion.embedder import Embedder
from src.ingestion.indexer import Indexer
from src.retrieval.dense import DenseRetriever
from src.retrieval.fusion import rrf_merge
from src.retrieval.reranker import Reranker
from src.retrieval.sparse import BM25Retriever


class Retriever:
    def __init__(self, index_path: str = "data/index", embed_model: str = "all-MiniLM-L6-v2"):
        self.indexer = Indexer(index_path=index_path)
        self.index, self.meta, self.bm25 = self.indexer.load()
        self.embedder = Embedder(embed_model)
        self.dense = DenseRetriever(self.index, self.meta)
        self.sparse = BM25Retriever(self.bm25, self.meta)
        self.reranker = Reranker(embed_model)
        self.last_confidence = 1.0

    def _query_embedding(self, query: str):
        try:
            return self.embedder.embed_query(query)
        except Exception:
            return None

    def retrieve(self, query: str, topk: int = 10, domain: Optional[str] = None) -> List[Dict]:
        sparse_res = self.sparse.retrieve(query, topk=topk)
        if domain:
            sparse_res = [item for item in sparse_res if item["meta"].get("domain") == domain]

        dense_res = []
        query_embedding = self._query_embedding(query)
        if query_embedding is not None:
            dense_res = self.dense.retrieve(query_embedding, topk=topk)
            if domain:
                dense_res = [item for item in dense_res if item["meta"].get("domain") == domain]

        fused = rrf_merge([sparse_res, dense_res], k=topk)
        reranked = self.reranker.rerank(query, fused, topk=topk)

        if reranked:
            self.last_confidence = float(sum(item["score"] for item in reranked) / len(reranked))
        else:
            self.last_confidence = 0.0
        return reranked
