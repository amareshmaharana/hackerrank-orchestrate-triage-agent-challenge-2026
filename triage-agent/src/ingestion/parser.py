"""Simple file loaders: TXT, MD, HTML, and PDF.

Each document returned as dict with `source`, `title`, `domain`, and `text`.
PDF support prefers `pypdf` and falls back to a plain-text extraction attempt when available.
"""
from typing import List, Dict, Optional
import os
import glob
import re

try:
    from pypdf import PdfReader
except Exception:  # pragma: no cover - optional dependency
    PdfReader = None

def infer_domain_from_path(path: str) -> str:
    parts = path.replace("\\", "/").split("/")
    for p in parts:
        if p.lower() in ("hackerrank", "claude_help", "visa"):
            return p.lower()
    return "unknown"

def load_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def clean_html(html: str) -> str:
    text = re.sub(r"<script.*?>.*?</script>", "", html, flags=re.S)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_pdf_file(path: str, reader_cls: Optional[type] = None) -> str:
    """Extract text from a PDF file.

    reader_cls is injectable for tests. When omitted, uses pypdf.PdfReader if available.
    """
    if reader_cls is None:
        reader_cls = PdfReader
    if reader_cls is None:
        # Minimal fallback parser for simple text-based PDFs. This handles the
        # generated fixture and many basic support documents without extra deps.
        with open(path, "rb") as f:
            raw = f.read().decode("latin-1", errors="ignore")
        # Prefer content inside stream objects.
        stream_chunks = re.findall(r"stream\s*(.*?)\s*endstream", raw, flags=re.S)
        search_space = "\n".join(stream_chunks) if stream_chunks else raw
        # Extract visible text from text-show operators.
        text_fragments = re.findall(r"\((.*?)\)\s*Tj", search_space, flags=re.S)
        if not text_fragments:
            text_fragments = re.findall(r"\((.*?)\)", search_space, flags=re.S)
        return clean_html("\n".join(text_fragments))

    text_parts = []
    with open(path, "rb") as f:
        reader = reader_cls(f)
        pages = getattr(reader, "pages", [])
        for page in pages:
            page_text = ""
            try:
                page_text = page.extract_text() or ""
            except Exception:
                page_text = ""
            if page_text:
                text_parts.append(page_text)
    return clean_html("\n".join(text_parts))

def load_documents_from_dir(root: str) -> List[Dict]:
    root = os.path.abspath(root)
    docs = []
    patterns = ["**/*.txt", "**/*.md", "**/*.html", "**/*.pdf"]
    for pat in patterns:
        for path in glob.glob(os.path.join(root, pat), recursive=True):
            try:
                if path.lower().endswith(".pdf"):
                    text = load_pdf_file(path)
                else:
                    text = load_text_file(path)
                if path.lower().endswith(".html"):
                    text = clean_html(text)
                domain = infer_domain_from_path(path)
                title = os.path.splitext(os.path.basename(path))[0]
                docs.append({"source": path, "title": title, "domain": domain, "text": text})
            except Exception:
                continue
    return docs
