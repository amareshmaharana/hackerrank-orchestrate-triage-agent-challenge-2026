from src.ingestion.parser import load_pdf_file


class _FakePage:
    def __init__(self, text: str):
        self._text = text

    def extract_text(self):
        return self._text


class _FakePdfReader:
    def __init__(self, file_obj):
        self.pages = [_FakePage("First page text."), _FakePage("Second page text.")]


def test_load_pdf_file_uses_reader_and_concatenates_pages(tmp_path):
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake content")

    text = load_pdf_file(str(pdf_path), reader_cls=_FakePdfReader)

    assert "First page text" in text
    assert "Second page text" in text


def _build_minimal_text_pdf_bytes(text: str) -> bytes:
    def obj(num: int, body: str) -> bytes:
        return f"{num} 0 obj\n{body}\nendobj\n".encode("latin-1")

    content_stream = f"BT /F1 24 Tf 72 100 Td ({text}) Tj ET"
    objects = []
    objects.append(b"%PDF-1.4\n")

    offsets = [0]
    current = len(objects[0])

    parts = [
        obj(1, "<< /Type /Catalog /Pages 2 0 R >>"),
        obj(2, "<< /Type /Pages /Kids [3 0 R] /Count 1 >>"),
        obj(3, "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 144] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>"),
        obj(4, "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"),
        f"5 0 obj\n<< /Length {len(content_stream.encode('latin-1'))} >>\nstream\n{content_stream}\nendstream\nendobj\n".encode("latin-1"),
    ]

    for part in parts:
        offsets.append(current)
        current += len(part)
        objects.append(part)

    xref_start = current
    xref = [b"xref\n", b"0 6\n", b"0000000000 65535 f \n"]
    for off in offsets[1:]:
        xref.append(f"{off:010d} 00000 n \n".encode("latin-1"))
    trailer = f"trailer\n<< /Root 1 0 R /Size 6 >>\nstartxref\n{xref_start}\n%%EOF\n".encode("latin-1")

    return b"".join(objects + xref + [trailer])


def test_load_pdf_file_reads_real_generated_pdf(tmp_path):
    pdf_path = tmp_path / "generated.pdf"
    pdf_path.write_bytes(_build_minimal_text_pdf_bytes("Generated PDF fixture text"))

    text = load_pdf_file(str(pdf_path))

    assert "Generated PDF fixture text" in text
