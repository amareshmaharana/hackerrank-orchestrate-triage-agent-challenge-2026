from src.generation.response_generator import generate_response


def test_generate_grounded_response():
    query = "How do I resubmit a challenge"
    # make a fake retrieved chunk that contains supporting text
    chunk = {"score": 1.0, "meta": {"chunk_id": "c1", "text": "To resubmit a challenge, navigate to your dashboard and click Submit again. This will reopen the submission window.", "title": "Resubmit", "domain": "hackerrank"}}
    decision = {"action": "answer", "score": 0.9}
    classification = {"domain": "hackerrank"}
    out = generate_response(query, [chunk], classification, decision)
    assert out['grounded'] is True
    assert 'resubmit' in out['response'].lower()


def test_generate_refusal_when_not_grounded():
    query = "What is my refund status for credit card 4111 1111 1111 1111"
    # retrieved chunk unrelated
    chunk = {"score": 0.1, "meta": {"chunk_id": "c2", "text": "General billing FAQs and how to view invoices.", "title": "Billing", "domain": "visa"}}
    decision = {"action": "answer", "score": 0.2}
    classification = {"domain": "visa"}
    out = generate_response(query, [chunk], classification, decision)
    # should refuse because grounding is insufficient or PII detected in query
    assert 'cannot' in out['response'].lower() or 'escalate' in out['response'].lower() or out['grounded'] is False
