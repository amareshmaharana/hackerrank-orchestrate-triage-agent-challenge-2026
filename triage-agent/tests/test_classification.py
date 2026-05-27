from src.classification.domain_classifier import classify_domain
from src.classification.intent_classifier import classify_intent
from src.classification.urgency_scorer import score_urgency


def test_classify_domain_uses_keywords_when_metadata_missing():
    text = "I cannot submit my HackerRank challenge because the score did not update."

    assert classify_domain(text) == "hackerrank"


def test_classify_domain_prefers_metadata():
    text = "This message mentions visa but metadata should win."

    assert classify_domain(text, metadata={"domain": "claude_help"}) == "claude_help"


def test_classify_intent_detects_billing_issue():
    text = "I need a refund for an unexpected charge on my invoice."

    assert classify_intent(text) == "billing_issue"


def test_classify_intent_detects_account_access():
    text = "I cannot sign in because my password was reset and the account is locked."

    assert classify_intent(text) == "account_access"


def test_score_urgency_matches_keywords():
    assert score_urgency("This is urgent and I need help now") == "high"
    assert score_urgency("Please look at this soon") == "medium"
    assert score_urgency("No rush, just a question") == "low"
