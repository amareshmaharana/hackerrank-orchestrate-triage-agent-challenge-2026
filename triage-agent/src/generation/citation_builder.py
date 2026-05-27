"""Build citation lists for presented chunks."""
from typing import List, Dict

def build_citations(chunks: List[Dict]) -> str:
    lines = []
    for c in chunks:
        meta = c.get('meta') if 'meta' in c else c
        lines.append(f"- {meta.get('title')} | {meta.get('source')} | {meta.get('section')}")
    return "\n".join(lines)
