"""Deeper evaluation helpers for hallucination detection and latency logging."""
import time
from typing import Callable, List, Dict

def timeit(func: Callable):
    def wrapper(*args, **kwargs):
        start = time.time()
        res = func(*args, **kwargs)
        return {"result": res, "latency": time.time() - start}
    return wrapper

def detect_hallucination(answer: str, retrieved_texts: List[str]) -> bool:
    # simple heuristic: does answer include facts not appearing in retrieved_texts?
    corpus = "\n".join(retrieved_texts).lower()
    for sentence in answer.split('. '):
        s = sentence.strip().lower()
        if len(s) < 10:
            continue
        # if sentence has unique token not in corpus, flag
        tokens = [t for t in s.split() if len(t) > 4]
        if tokens and not any(t in corpus for t in tokens[:3]):
            return True
    return False
