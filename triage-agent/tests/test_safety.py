from src.safety.pii_detector import detect_pii
from src.safety.risk_scorer import compute_risk
from src.safety.escalation_engine import decide


def test_detect_pii_finds_email_ssn_and_card():
    text = "Contact me at jane.doe@example.com. SSN 123-45-6789. Card 4111 1111 1111 1111."
    pii = detect_pii(text)

    assert "email" in pii
    assert "ssn" in pii
    assert "credit_card" in pii


def test_compute_risk_increases_for_pii_and_fraud_keywords():
    text = "This is fraud. My email is jane.doe@example.com and my card is 4111 1111 1111 1111."
    risk = compute_risk(text)

    assert risk >= 1.0 or risk == 1.0


def test_escalation_on_pii_and_low_confidence():
    decision = decide(risk_score=0.2, confidence=0.3, has_pii=True)

    assert decision.action == "escalate"
    assert decision.reason == "PII present"


def test_escalation_on_high_risk_even_without_pii():
    decision = decide(risk_score=0.8, confidence=0.9, has_pii=False)

    assert decision.action == "escalate"
    assert decision.reason == "High risk content"


def test_refuse_on_medium_risk_when_confident():
    decision = decide(risk_score=0.5, confidence=0.9, has_pii=False)

    assert decision.action == "refuse"
    assert decision.reason == "Potentially risky"
