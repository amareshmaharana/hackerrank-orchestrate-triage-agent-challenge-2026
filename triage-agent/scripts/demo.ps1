param(
    [string]$Query = "How do I resubmit a challenge?",
    [string]$IndexPath = "data/index"
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
Set-Location ..

$requiredPackages = @("numpy", "sentence_transformers", "faiss", "rank_bm25", "rich", "typer", "pydantic")
foreach ($package in $requiredPackages) {
    $check = py -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('$package') else 1)" 2>$null
    if ($LASTEXITCODE -ne 0) {
        throw "Missing Python dependency: $package. Install the project dependencies first, then rerun scripts/demo.ps1."
    }
}

Write-Host "[1/3] Building index..."
py scripts/build_index.py

Write-Host "[2/3] Running smoke triage..."
@'
import json
import os
import sys

sys.path.insert(0, '.')

from src.classification.domain_classifier import classify_domain
from src.classification.intent_classifier import classify_intent
from src.classification.urgency_scorer import score_urgency
from src.generation.response_generator import generate_response
from src.logging_.csv_exporter import CSVExporter
from src.logging_.transcript_logger import TranscriptLogger
from src.retrieval.retriever import Retriever
from src.safety.escalation_engine import decide
from src.safety.pii_detector import detect_pii
from src.safety.risk_scorer import compute_risk

query = os.environ.get("TRIAGE_QUERY", "How do I resubmit a challenge?")
index_path = os.environ.get("TRIAGE_INDEX_PATH", "data/index")

retriever = Retriever(index_path=index_path)
retrieved = retriever.retrieve(query, topk=5)

domain = classify_domain(query)
intent = classify_intent(query)
urgency = score_urgency(query)
pii = detect_pii(query)
risk = compute_risk(query)
retrieval_confidence = getattr(retriever, "last_confidence", 0.0)
decision = decide(risk_score=risk, confidence=retrieval_confidence, has_pii=bool(pii))
response = generate_response(query, retrieved, {"domain": domain, "intent": intent, "pii": pii}, decision.__dict__)

logger = TranscriptLogger()
logger.log({
    "query": query,
    "domain": domain,
    "intent": intent,
    "urgency": urgency,
    "pii": pii,
    "risk": risk,
    "retrieval_confidence": retrieval_confidence,
    "decision": decision.__dict__,
    "retrieved": retrieved,
    "response": response,
    "mode": "demo_smoke",
})

exporter = CSVExporter()
exporter.export([{
    "ticket_id": "demo-smoke",
    "request_type": intent,
    "product_area": domain,
    "risk_level": "high" if risk > 0.6 else ("medium" if risk > 0.2 else "low"),
    "escalation": decision.action,
    "retrieved_sources": ";".join([r["meta"]["chunk_id"] for r in retrieved]),
    "final_response": response["response"],
    "confidence_score": decision.score,
}])

print(json.dumps({
    "query": query,
    "domain": domain,
    "intent": intent,
    "urgency": urgency,
    "risk": risk,
    "retrieval_confidence": retrieval_confidence,
    "decision": decision.__dict__,
    "grounded": response.get("grounded"),
    "grounding_score": response.get("grounding_score"),
    "response": response["response"],
}, indent=2))
print(f"Transcript: {logger.path}")
print(f"CSV: {exporter.out_path}")

if not retrieved:
    raise SystemExit("Smoke test failed: no retrieval results returned.")
'@ | py -

Write-Host "[3/3] Done."
