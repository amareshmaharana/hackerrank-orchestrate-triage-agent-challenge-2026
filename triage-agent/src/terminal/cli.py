import typer
from src.terminal.display import show_ticket, show_retrieved, show_decision, show_classification
from src.logging_.transcript_logger import TranscriptLogger
from src.retrieval.retriever import Retriever
from src.classification.domain_classifier import classify_domain
from src.classification.intent_classifier import classify_intent
from src.classification.urgency_scorer import score_urgency
from src.safety.pii_detector import detect_pii
from src.safety.risk_scorer import compute_risk
from src.safety.escalation_engine import decide
from src.generation.response_generator import generate_response
from src.logging_.csv_exporter import CSVExporter
import uuid
import os

app = typer.Typer()

@app.command()
def start(index_path: str = "data/index"):
    """Start interactive triage session."""
    retriever = Retriever(index_path=index_path)
    logger = TranscriptLogger()
    exporter = CSVExporter()
    rows = []
    typer.echo("Triage Agent started. Type 'quit' to exit.")
    while True:
        q = typer.prompt("Enter ticket text or 'quit'")
        if q.strip().lower() in ("quit","exit"):
            break
        ticket_id = str(uuid.uuid4())
        ticket = {"ticket_id": ticket_id, "text": q}
        show_ticket(ticket)
        # classify
        domain = classify_domain(q)
        intent = classify_intent(q)
        urgency = score_urgency(q)
        pii = detect_pii(q)
        risk = compute_risk(q)
        # retrieval
        retrieved = retriever.retrieve(q, topk=5)
        show_retrieved(retrieved)
        # generation
        retrieval_confidence = getattr(retriever, 'last_confidence', 0.0)
        decision = decide(risk_score=risk, confidence=retrieval_confidence, has_pii=bool(pii))
        show_classification({
            "domain": domain,
            "intent": intent,
            "urgency": urgency,
            "risk": risk,
            "retrieval_confidence": retrieval_confidence,
            "action": decision.action,
        })
        show_decision(decision.__dict__)
        response = generate_response(q, retrieved, {"domain":domain,"intent":intent}, decision.__dict__)
        typer.echo(response['response'])
        # logging
        log = {
            "ticket_id": ticket_id,
            "text": q,
            "classification": {
                "domain": domain,
                "intent": intent,
                "urgency": urgency,
            },
            "domain": domain,
            "intent": intent,
            "urgency": urgency,
            "pii": pii,
            "risk": risk,
            "retrieval_confidence": retrieval_confidence,
            "decision": decision.__dict__,
            "response": response,
            "retrieved": retrieved
        }
        logger.log(log)
        rows.append({
            "ticket_id": ticket_id,
            "request_type": intent,
            "product_area": domain,
            "risk_level": "high" if risk>0.6 else ("medium" if risk>0.2 else "low"),
            "escalation": decision.action,
            "retrieved_sources": ";".join([r['meta']['chunk_id'] for r in retrieved]),
            "final_response": response['response'],
            "confidence_score": decision.score
        })
    exporter.export(rows)
    typer.echo(f"Session saved: {logger.path}, CSV: {exporter.out_path}")
