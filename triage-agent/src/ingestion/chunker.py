"""Chunking utilities: recursive and semantic-friendly chunking with overlap.
Produces list of chunk dicts with required metadata.
"""
from typing import List, Dict
import uuid

DEFAULT_CHUNK_SIZE = 800
DEFAULT_OVERLAP = 100

def text_chunks(text: str, size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_OVERLAP) -> List[str]:
    tokens = text.split()
    if len(tokens) <= size:
        return [" ".join(tokens)]
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + size, len(tokens))
        chunks.append(" ".join(tokens[start:end]))
        if end == len(tokens):
            break
        start = end - overlap
    return chunks

def chunk_documents(docs: List[Dict], chunk_size: int = DEFAULT_CHUNK_SIZE, chunk_overlap: int = DEFAULT_OVERLAP) -> List[Dict]:
    out = []
    for d in docs:
        chunks = text_chunks(d.get("text", ""), size=chunk_size, overlap=chunk_overlap)
        for i, c in enumerate(chunks):
            out.append({
                "chunk_id": str(uuid.uuid4()),
                "source": d.get("source"),
                "domain": d.get("domain"),
                "title": d.get("title"),
                "section": f"chunk_{i}",
                "text": c
            })
    return out
