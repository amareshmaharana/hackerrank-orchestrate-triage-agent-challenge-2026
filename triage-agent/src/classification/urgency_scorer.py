"""Simple urgency heuristics based on keywords and presence of time words."""
from typing import Literal

def score_urgency(text: str) -> Literal["low","medium","high"]:
    t = text.lower()
    if any(w in t for w in ["asap","urgent","immediately","critical","now"]):
        return "high"
    if any(w in t for w in ["soon","priority","important"]):
        return "medium"
    return "low"
