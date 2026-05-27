
Triager — Multi-Domain Support Triage Agent
==========================================

Triager is a terminal-first support triage system for three offline corpora:
HackerRank Support, Claude Help Center, and Visa Support. It classifies the
incoming ticket, retrieves only from the local knowledge base, and either
answers with grounded citations or escalates when the evidence is weak or the
request looks risky.

The project is designed for the HackerRank Orchestrate challenge, but the
components are intentionally modular so the same pipeline can be used for
manual demos, batch evaluation, and future production hardening.

What this project does
----------------------

- Uses only the supplied support corpus. No internet calls and no external knowledge.
- Combines sparse BM25 and dense FAISS retrieval, then fuses and reranks results.
- Applies deterministic classification, PII detection, and risk-based escalation.
- Generates conservative responses from retrieved text only, with citations.
- Exports structured CSV results and JSONL transcripts for auditing and review.

Repository layout
-----------------

- [configs/](configs/) contains the global and domain YAML settings.
- [data/raw/](data/raw/) is where domain corpora are placed before indexing.
- [data/index/](data/index/) stores the generated FAISS, BM25, and metadata artifacts.
- [data/eval/](data/eval/) contains the sample evaluation CSV.
- [scripts/](scripts/) holds the runnable entry points.
- [src/](src/) contains the ingestion, retrieval, classification, safety, generation, logging, terminal, and evaluation modules.
- [outputs/](outputs/) stores exported CSVs and transcript logs.

Project configuration
---------------------

The main configuration files are:

- [configs/base.yaml](configs/base.yaml) for shared settings such as the embedding model, retrieval top-k values, and escalation thresholds.
- [configs/hackerrank.yaml](configs/hackerrank.yaml) for HackerRank corpus chunking settings.
- [configs/claude_help.yaml](configs/claude_help.yaml) for Claude Help corpus chunking settings.
- [configs/visa.yaml](configs/visa.yaml) for Visa corpus chunking settings.

How to install
--------------

Create a virtual environment, activate it, then install the dependencies from
the included requirements file:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirement.txt
```

If you prefer a different environment manager, install the same packages listed
in [requirement.txt](requirement.txt).

Required data layout
--------------------

Place source documents in domain folders under [data/raw/](data/raw/):

- [data/raw/hackerrank/](data/raw/hackerrank/)
- [data/raw/claude_help/](data/raw/claude_help/)
- [data/raw/visa/](data/raw/visa/)

Supported inputs are `.txt`, `.md`, and `.html`. PDF support is possible by
extending the parser and using a PDF extraction library.

Run the project
---------------

1. Build the local index from the raw corpus:

```bash
python scripts/build_index.py
```

This reads [data/raw/](data/raw/), chunks the documents, generates embeddings,
and writes the persisted artifacts to [data/index/](data/index/).

2. Start the interactive terminal agent:

```bash
python scripts/run_agent.py
```

The REPL prompts for ticket text, shows the classification and retrieval
results, and logs the session to [outputs/transcripts/](outputs/transcripts/)
and [outputs/csv/](outputs/csv/).

3. Run the batch submission pipeline:

```bash
python scripts/batch_eval.py --input data/eval/support_tickets.csv --index data/index --output outputs/csv/output.csv
```

This is the main end-to-end evaluation path for the challenge dataset.

Environment variables
---------------------

The scripts recognize a small set of optional environment variables:

- `DATA_ROOT` defaults to `data/raw`
- `INDEX_ROOT` defaults to `data/index`
- `EMBED_MODEL` defaults to `all-MiniLM-L6-v2`

Core workflow
-------------

1. Ingestion in [src/ingestion/](src/ingestion/) reads documents from the raw corpus and attaches metadata such as source, title, and domain.
2. Chunking splits long documents into overlapping segments so retrieval stays precise.
3. Embedding uses `sentence-transformers` to produce dense vectors for each chunk.
4. Indexing persists the FAISS index, BM25 corpus, and chunk metadata.
5. Retrieval in [src/retrieval/](src/retrieval/) combines sparse and dense search, fuses the results, and reranks them.
6. Classification in [src/classification/](src/classification/) determines domain, intent, and urgency.
7. Safety in [src/safety/](src/safety/) checks for PII, computes risk, and chooses answer, refuse, or escalate.
8. Generation in [src/generation/](src/generation/) builds the final response only from retrieved text and applies grounding checks.
9. Logging in [src/logging_/](src/logging_) writes JSONL transcripts and CSV summaries for auditing.

Safety model
------------

The agent is intentionally conservative:

- PII detection uses regex heuristics for common sensitive patterns such as emails and payment-like numbers.
- Risk scoring combines the content of the ticket with safety signals.
- Escalation is deterministic when confidence is low, evidence is missing, or the request appears sensitive.
- Responses are not allowed to invent facts; if grounding is weak, the system escalates instead of guessing.

Outputs
-------

- [outputs/csv/output.csv](outputs/csv/output.csv) contains the structured batch evaluation result.
- [outputs/transcripts/](outputs/transcripts/) contains the detailed JSONL transcript logs.
- [data/index/](data/index/) contains the generated retrieval artifacts.

Testing and evaluation
----------------------

The repository includes unit and integration-style tests under [tests/](tests/).
To run them:

```bash
pytest
```

For more targeted validation, use the batch pipeline in [scripts/batch_eval.py](scripts/batch_eval.py) and inspect the exported CSV plus transcript logs.

Extending the project
---------------------

- Swap in a stronger reranker in [src/retrieval/reranker.py](src/retrieval/reranker.py) if you want better ranking quality.
- Improve parsing in [src/ingestion/parser.py](src/ingestion/parser.py) if you need more file formats.
- Replace FAISS with another vector backend if your corpus grows beyond the demo scale.
- Tighten the safety heuristics in [src/safety/](src/safety/) if you need stricter compliance behavior.

Limitations
-----------

- The current generation path is template-driven and does not call an external LLM.
- PII and risk checks are heuristic-based and should be replaced or augmented for production use.
- The current corpus support is optimized for the three challenge domains and may need more normalization for other support data.
