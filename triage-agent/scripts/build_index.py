"""Run offline ingestion -> embedding -> indexing pipeline."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ingestion.parser import load_documents_from_dir
from src.ingestion.chunker import chunk_documents
from src.ingestion.embedder import Embedder
from src.ingestion.indexer import Indexer

DATA_ROOT = os.getenv("DATA_ROOT", "data/raw")
INDEX_ROOT = os.getenv("INDEX_ROOT", "data/index")

def main():
    # load raw docs
    docs = load_documents_from_dir(DATA_ROOT)
    # chunk
    chunks = chunk_documents(docs)
    # embed
    embedder = Embedder(model_name=os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2"))
    embeddings = embedder.embed_chunks(chunks)
    # index
    indexer = Indexer(index_path=INDEX_ROOT)
    indexer.build_index(chunks, embeddings)
    print(f"Indexed {len(chunks)} chunks to {INDEX_ROOT}")

if __name__ == "__main__":
    main()
