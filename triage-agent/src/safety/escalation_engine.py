"""Deterministic escalation engine: decide answer/refuse/escalate based on scores and rules."""
from typing import Dict
from dataclasses import dataclass

@dataclass
class EscalationDecision:
    action: str  # 'answer'|'escalate'|'refuse'
    reason: str
    score: float

def decide(risk_score: float, confidence: float, has_pii: bool) -> EscalationDecision:
    # deterministic rules
    if has_pii and risk_score > 0.0:
        return EscalationDecision(action='escalate', reason='PII present', score=risk_score)
    if risk_score >= 0.7:
        return EscalationDecision(action='escalate', reason='High risk content', score=risk_score)
    if confidence < 0.5:
        return EscalationDecision(action='escalate', reason='Low confidence', score=confidence)
    if risk_score > 0.4:
        return EscalationDecision(action='refuse', reason='Potentially risky', score=risk_score)
    return EscalationDecision(action='answer', reason='Safe to answer', score=1.0 - risk_score)
