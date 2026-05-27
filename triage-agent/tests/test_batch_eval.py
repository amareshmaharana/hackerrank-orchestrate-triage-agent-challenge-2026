import csv
from pathlib import Path

import pytest

from scripts import batch_eval


class DummyRetriever:
    def __init__(self, index_path=None):
        self.last_confidence = 0.0

    def retrieve(self, query, topk=5, domain=None):
        return []


class DummyTranscript:
    def __init__(self, *a, **k):
        pass

    def log(self, *a, **k):
        pass


def test_batch_eval_smoke(tmp_path, monkeypatch):
    # Create a minimal input CSV
    input_csv = tmp_path / "tickets.csv"
    rows = [
        {"Issue": "Test issue 1", "Subject": "S1", "Company": "HackerRank"},
        {"Issue": "Test issue 2", "Subject": "S2", "Company": "Visa"},
    ]
    with input_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Issue", "Subject", "Company"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    out_csv = tmp_path / "out.csv"

    # Monkeypatch heavy components to keep the test lightweight
    monkeypatch.setattr(batch_eval, "Retriever", DummyRetriever)
    monkeypatch.setattr(batch_eval, "TranscriptLogger", DummyTranscript)

    result_rows = batch_eval.run(cases_path=str(input_csv), index_path=str(tmp_path / "index"), out_path=str(out_csv))

    assert len(result_rows) == 2
    assert out_csv.exists()

    with out_csv.open("r", encoding="utf-8") as f:
        header = f.readline().strip().split(",")

    expected = [
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
    assert header == expected
