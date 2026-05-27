"""Intent classifier using keyword heuristics and fallback similarity.
"""
from typing import Dict
INTENT_KEYWORDS = {
    "billing_issue": ["invoice","charge","billing","refund","payment"],
    "account_access": ["login","password","sign in","locked","2fa","two-factor"],
    "technical_issue": ["error","bug","crash","stack trace","failed"],
    "general_query": ["how do i","how to","what is","where can i"]
}

def classify_intent(text: str) -> str:
    t = text.lower()
    scores = {k:0 for k in INTENT_KEYWORDS}
    for k,kws in INTENT_KEYWORDS.items():
        for kw in kws:
            if kw in t:
                scores[k] += 1
    best = max(scores.items(), key=lambda x: x[1])
    if best[1] == 0:
        return "other"
    return best[0]
