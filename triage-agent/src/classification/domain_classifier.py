"""Domain classifier based on simple keyword heuristics and path metadata.
"""
from typing import Dict

DOMAIN_KEYWORDS = {
    "hackerrank": ["submission","challenge","score","hackerrank","problem"],
    "claude_help": ["claude","anthropic","help center","faq","claude help"],
    "visa": ["visa","application","passport","immigration","consulate"]
}

def classify_domain(text: str, metadata: Dict = None) -> str:
    if metadata and metadata.get("domain"):
        return metadata.get("domain")
    t = text.lower()
    scores = {k:0 for k in DOMAIN_KEYWORDS}
    for k, kws in DOMAIN_KEYWORDS.items():
        for kw in kws:
            if kw in t:
                scores[k] += 1
    best = max(scores.items(), key=lambda x: x[1])
    if best[1] == 0:
        return "unknown"
    return best[0]
