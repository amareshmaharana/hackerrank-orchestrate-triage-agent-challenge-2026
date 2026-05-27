"""Grounding verifier with explainable overlap scoring.

This module computes a grounding score as the proportion of answer n-grams
that are present in the retrieved chunk corpus. Returns a tuple
(score: float, grounded: bool) using a conservative threshold.
"""
from typing import List, Dict, Tuple
import re


def _tokenize(s: str) -> List[str]:
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    toks = [t for t in s.split() if t]
    return toks


def _ngrams(tokens: List[str], n: int = 3) -> List[str]:
    if len(tokens) < n:
        return [' '.join(tokens)] if tokens else []
    return [' '.join(tokens[i:i+n]) for i in range(0, len(tokens) - n + 1)]


def verify_grounding(answer: str, chunks: List[Dict], ngram_n: int = 3, threshold: float = 0.12) -> Tuple[float, bool]:
    """Return (score, grounded).

    score is fraction of answer ngrams found in the retrieved corpus.
    grounded is True if score >= threshold.
    """
    ans_toks = _tokenize(answer)
    if not ans_toks:
        return 0.0, False
    ans_ngrams = _ngrams(ans_toks, n=ngram_n)

    corpus = " \n ".join([c['meta']['text'] if 'meta' in c else c.get('text','') for c in chunks]).lower()
    corpus_toks = _tokenize(corpus)
    corpus_ngrams = set(_ngrams(corpus_toks, n=ngram_n))

    if not ans_ngrams:
        return 0.0, False
    hits = sum(1 for ng in ans_ngrams if ng in corpus_ngrams)
    score = hits / len(ans_ngrams)
    grounded = score >= threshold
    return float(score), bool(grounded)

