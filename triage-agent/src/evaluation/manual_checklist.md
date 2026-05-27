Evaluation Manual Checklist
=========================

- Verify indexing: FAISS index file exists and BM25 corpus present.
- Run sample queries and confirm top-1 is relevant.
- Validate PII detection on known PII examples.
- Test escalation rules with crafted risky messages.
- Run latency benchmarks for embedding + retrieval.
- Check CSV/export and transcript formats.
