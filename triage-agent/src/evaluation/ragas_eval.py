"""RAG-as-evaluation utilities: precision@k, recall@k, grounding checks."""
from typing import List, Dict

def precision_at_k(retrieved: List[Dict], relevant_ids: List[str], k: int = 5) -> float:
    topk = retrieved[:k]
    if not topk:
        return 0.0
    hits = sum(1 for r in topk if r['meta']['chunk_id'] in relevant_ids)
    return hits / len(topk)

def recall_at_k(retrieved: List[Dict], relevant_ids: List[str], k: int = 5) -> float:
    topk = retrieved[:k]
    if not relevant_ids:
        return 0.0
    hits = sum(1 for r in topk if r['meta']['chunk_id'] in relevant_ids)
    return hits / len(relevant_ids)

def grounding_rate(responses: List[Dict]) -> float:
    # responses: list of {'grounded': bool}
    if not responses:
        return 0.0
    grounded = sum(1 for r in responses if r.get('grounded'))
    return grounded / len(responses)
