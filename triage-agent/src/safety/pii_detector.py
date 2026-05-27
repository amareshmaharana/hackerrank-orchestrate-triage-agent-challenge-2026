"""PII detector using regex heuristics."""
import re
from typing import Dict

PII_PATTERNS = {
    'email': re.compile(r"[\w\.-]+@[\w\.-]+"),
    'ssn': re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    'credit_card': re.compile(r"\b(?:\d[ -]*?){13,16}\b")
}

def detect_pii(text: str) -> Dict[str, list]:
    found = {}
    for k, pat in PII_PATTERNS.items():
        matches = pat.findall(text)
        if matches:
            found[k] = matches
    return found
