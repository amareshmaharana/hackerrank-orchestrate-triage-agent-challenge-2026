import os
from pathlib import Path
from src.ingestion.parser import load_documents_from_dir
from src.ingestion.chunker import chunk_documents
from src.ingestion.embedder import Embedder
from src.ingestion.indexer import Indexer
from src.retrieval.retriever import Retriever


def test_end_to_end_index_and_retrieve(tmp_path):
    # prepare paths
    project_root = Path(__file__).resolve().parents[1]
    data_root = project_root / 'data' / 'raw'
    index_dir = tmp_path / "index"
    index_dir = str(index_dir)

    docs = load_documents_from_dir(str(data_root))
    assert len(docs) > 0, "No docs found in data/raw for test"
    chunks = chunk_documents(docs)
    assert len(chunks) > 0

    embedder = Embedder()
    embeddings = embedder.embed_chunks(chunks, batch_size=8)

    indexer = Indexer(index_path=index_dir)
    indexer.build_index(chunks, embeddings)

    retriever = Retriever(index_path=index_dir)
    res = retriever.retrieve("resubmit challenge", topk=5)
    assert isinstance(res, list)
    assert len(res) > 0, "Retriever returned no results"
