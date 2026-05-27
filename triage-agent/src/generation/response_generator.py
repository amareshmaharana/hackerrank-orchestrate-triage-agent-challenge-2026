"""Generates grounded responses using only retrieved chunks.
This implementation uses template-based assembly to avoid hallucinations.
"""
from typing import List, Dict
from src.generation.citation_builder import build_citations
from src.generation.grounding_verifier import verify_grounding
from src.safety.pii_detector import detect_pii
import os


def _load_template(name: str) -> str:
    path = os.path.join(os.path.dirname(__file__), "prompts", name)
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception:
        return ""


def generate_response(query: str, retrieved: List[Dict], classification: Dict, decision: Dict) -> Dict:
    """Assemble a safe, grounded response or return a refusal/escalation template.

    The generator only uses retrieved content. It computes grounding score and
    will refuse/escalate if grounding is insufficient or safety decision requires it.
    """
    if not retrieved:
        tmpl = _load_template('refusal.txt')
        return {"response": tmpl or "No documentation found; escalating to human.", "citations": [], "grounded": False, "grounding_score": 0.0}

    # Guardrail: never provide a normal answer when user query contains PII.
    if detect_pii(query):
        tmpl = _load_template('refusal.txt')
        return {
            "response": tmpl or "I cannot process sensitive personal data in this channel. Escalating.",
            "citations": [],
            "grounded": False,
            "grounding_score": 0.0,
        }

    top = retrieved[:3]
    citations = build_citations(top)

    # assemble conservative answer by extracting sentences overlapping the query
    tokens = set([t for t in query.lower().split() if len(t) > 3])
    quoted = []
    for c in top:
        text = c['meta']['text'] if 'meta' in c else c.get('text','')
        for sent in text.split('. '):
            low = sent.lower()
            if any(t in low for t in tokens) and len(sent) > 30:
                quoted.append(sent.strip())
                break
    body = "\n\n".join(quoted) if quoted else (top[0]['meta']['text'][:600] if 'meta' in top[0] else top[0].get('text',''))
    response_text = f"Answer (grounded):\n\n{body}\n\nSources:\n{citations}"

    grounding_score, grounded = verify_grounding(response_text, top)

    # if safety decision is escalate or refuse, return templates
    if decision.get('action') == 'escalate':
        tmpl = _load_template('escalation.txt')
        formatted = tmpl.format(user_message=query, pii=classification.get('pii',''), risk=decision.get('score',0.0), confidence=decision.get('score',0.0), top_sources=citations)
        return {"response": formatted, "citations": citations, "grounded": False, "grounding_score": grounding_score}
    if decision.get('action') == 'refuse' or not grounded:
        tmpl = _load_template('refusal.txt')
        return {"response": tmpl or "I cannot answer based on available documentation. Escalating.", "citations": citations, "grounded": grounded, "grounding_score": grounding_score}

    return {"response": response_text, "citations": citations, "grounded": grounded, "grounding_score": grounding_score}

