"""Primary submission workflow for the HackerRank Orchestrate challenge.

This script loads `data/eval/support_tickets.csv`, runs the full triage
pipeline, writes `outputs/csv/output.csv`, and logs structured transcripts to
`outputs/transcripts/log.txt`.
"""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - optional dependency fallback
    def tqdm(iterable, **kwargs):
        return iterable

from src.classification.domain_classifier import classify_domain
from src.classification.intent_classifier import classify_intent
from src.classification.urgency_scorer import score_urgency
from src.evaluation.ragas_eval import precision_at_k, recall_at_k
from src.generation.response_generator import generate_response
from src.logging_.csv_exporter import CSVExporter
from src.logging_.transcript_logger import TranscriptLogger
from src.retrieval.retriever import Retriever
from src.safety.escalation_engine import decide
from src.safety.pii_detector import detect_pii
from src.safety.risk_scorer import compute_risk

DEFAULT_INPUT = "data/eval/support_tickets.csv"
DEFAULT_OUTPUT = "outputs/csv/output.csv"

OUTPUT_COLUMNS = [
    "ticket_id",
    "issue",
    "domain",
    "request_type",
    "urgency",
    "risk_score",
    "retrieval_confidence",
    "escalation",
    "retrieved_sources",
    "response",
]


def _normalize_header(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def _load_csv(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            normalized = { _normalize_header(k): (v or "").strip() for k, v in row.items() if k }
            rows.append(normalized)
        return rows


def _pick(*values: str) -> str:
    for value in values:
        if value and value.strip():
            return value.strip()
    return ""


_DOMAIN_COMPANY_MAP = {
    "hackerrank": "hackerrank",
    "claude": "claude_help",
    "claudehelp": "claude_help",
    "anthropic": "claude_help",
    "visa": "visa",
}


def _company_to_domain(company: str, issue_text: str) -> str:
    company_key = "".join(ch for ch in (company or "").lower() if ch.isalnum())
    if company_key in _DOMAIN_COMPANY_MAP:
        return _DOMAIN_COMPANY_MAP[company_key]
    inferred = classify_domain(issue_text)
    return inferred if inferred != "unknown" else "hackerrank"


def _ticket_query(issue: str, subject: str, company: str) -> str:
    parts = []
    if subject:
        parts.append(f"Subject: {subject}")
    if issue:
        parts.append(f"Issue: {issue}")
    if company:
        parts.append(f"Company: {company}")
    return "\n".join(parts).strip()


def process_tickets(cases_path: str = DEFAULT_INPUT, index_path: str = "data/index") -> List[Dict[str, str]]:
    tickets = _load_csv(cases_path)
    retriever = Retriever(index_path=index_path)
    transcript = TranscriptLogger()

    rows: List[Dict[str, str]] = []
    for idx, ticket in enumerate(tqdm(tickets, desc="Processing tickets"), start=1):
        issue = _pick(ticket.get("issue"), ticket.get("problem"), ticket.get("message"), ticket.get("text"))
        subject = _pick(ticket.get("subject"), ticket.get("title"), ticket.get("summary"))
        company = _pick(ticket.get("company"), ticket.get("domain"), ticket.get("product"))
        ticket_id = _pick(ticket.get("ticketid"), ticket.get("id"), str(idx)) or str(idx)

        query = _ticket_query(issue=issue, subject=subject, company=company)
        domain = _company_to_domain(company, f"{subject} {issue}")
        intent = classify_intent(query)
        urgency = score_urgency(query)
        pii = detect_pii(query)
        risk_score = compute_risk(query)

        retrieved = retriever.retrieve(query, topk=5, domain=domain if domain != "unknown" else None)
        retrieval_confidence = getattr(retriever, "last_confidence", 0.0)
        decision = decide(risk_score=risk_score, confidence=retrieval_confidence, has_pii=bool(pii))
        response = generate_response(query, retrieved, {"domain": domain, "intent": intent, "pii": pii}, decision.__dict__)

        precision5 = 0.0
        recall5 = 0.0
        if retrieved:
            precision5 = precision_at_k(retrieved, [r["meta"]["chunk_id"] for r in retrieved[:1]], k=5)
            recall5 = recall_at_k(retrieved, [r["meta"]["chunk_id"] for r in retrieved[:1]], k=5)

        transcript.log({
            "ticket_id": ticket_id,
            "issue": issue,
            "subject": subject,
            "company": company,
            "query": query,
            "domain": domain,
            "request_type": intent,
            "urgency": urgency,
            "pii": pii,
            "risk_score": risk_score,
            "retrieval_confidence": retrieval_confidence,
            "retrieved": retrieved,
            "escalation": decision.__dict__,
            "response": response,
        })

        rows.append({
            "ticket_id": ticket_id,
            "issue": issue,
            "subject": subject,
            "company": company,
            "domain": domain,
            "request_type": intent,
            "urgency": urgency,
            "risk_score": f"{risk_score:.4f}",
            "retrieval_confidence": f"{retrieval_confidence:.4f}",
            "escalation": decision.action,
            "escalation_reason": decision.reason,
            "retrieved_sources": ";".join(r["meta"]["chunk_id"] for r in retrieved),
            "response": response["response"],
            "grounded": str(bool(response.get("grounded"))),
            "grounding_score": f"{float(response.get('grounding_score', 0.0)):.4f}",
            "precision_at_5": f"{precision5:.4f}",
            "recall_at_5": f"{recall5:.4f}",
        })

    return rows


def run(cases_path: str = DEFAULT_INPUT, index_path: str = "data/index", out_path: str = DEFAULT_OUTPUT) -> List[Dict[str, str]]:
    rows = process_tickets(cases_path=cases_path, index_path=index_path)
    exporter = CSVExporter(out_path=out_path)
    exporter.export(rows, fieldnames=OUTPUT_COLUMNS)
    print(f"Batch eval complete. Wrote {len(rows)} rows to {out_path}")
    return rows


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Run the triage agent submission pipeline")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Path to support_tickets.csv")
    parser.add_argument("--index", default="data/index", help="Path to the built index directory")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Path to output.csv")
    args = parser.parse_args()

    run(cases_path=args.input, index_path=args.index, out_path=args.output)
