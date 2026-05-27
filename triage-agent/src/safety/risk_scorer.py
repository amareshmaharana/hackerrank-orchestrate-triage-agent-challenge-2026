"""Hybrid risk scoring combining deterministic rules and lightweight ML signals."""
from typing import Dict
from src.safety.pii_detector import detect_pii

def compute_risk(text: str, metadata: Dict = None) -> float:
    # base score
    score = 0.0
    # PII raises risk
    pii = detect_pii(text)
    if pii:
        score += 0.6
    # keywords
    t = text.lower()
    if any(k in t for k in ["fraud","scam","unauthorized","stolen","compromise"]):
        score += 0.4
    if any(k in t for k in ["refund","chargeback","billing dispute"]):
        score += 0.2
    return min(score, 1.0)
