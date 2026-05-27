"""Fusion strategies: Reciprocal Rank Fusion (RRF) to combine BM25 and dense ranks."""
from typing import List, Dict

def rrf_merge(lists: List[List[Dict]], k: int = 10, c: float = 60.0):
    # lists: each is list of {'score', 'meta'} in rank order
    scores = {}
    for lst in lists:
        for rank, item in enumerate(lst):
            key = item['meta']['chunk_id']
            scores.setdefault(key, {"meta": item['meta'], "rrf": 0.0})
            scores[key]['rrf'] += 1.0 / (c + rank + 1)
    merged = sorted(scores.values(), key=lambda x: x['rrf'], reverse=True)
    return [{"score": v['rrf'], "meta": v['meta']} for v in merged[:k]]
