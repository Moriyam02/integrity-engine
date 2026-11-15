"""Run with:  python -m pytest tests -q   (or just: python tests/test_stage1.py)"""
import sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chunking import chunk_document, clean_text
from src.extractors import ExtractionResult, Segment, extract_file


def test_zero_byte_pdf_is_skipped():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "empty.pdf"
        p.write_bytes(b"")
        r = extract_file(p)
        assert r.status == "skipped" and "empty" in r.warning


def test_garbage_pdf_does_not_crash():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "bad.pdf"
        p.write_bytes(b"this is not a pdf")
        assert extract_file(p).status == "skipped"


def test_unsupported_type():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "notes.txt"
        p.write_text("hello")
        assert extract_file(p).status == "skipped"


def test_docx_roundtrip():
    from docx import Document
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "a.docx"
        doc = Document()
        doc.add_paragraph("Vertical integration gives the firm a cost advantage.")
        doc.save(str(p))
        r = extract_file(p)
        assert r.status == "ok" and "Vertical" in r.segments[0].text


def test_chunking_overlap_and_pages():
    segs = [Segment(text=" ".join(f"w{i}" for i in range(100)), page=1),
            Segment(text=" ".join(f"x{i}" for i in range(100)), page=2)]
    chunks = chunk_document(ExtractionResult("d.pdf", "pdf", segments=segs), chunk_words=80, overlap=20)
    assert len(chunks) == 3
    assert chunks[0].page_start == 1 and chunks[-1].page_end == 2
    # last 20 words of chunk 0 are the first 20 words of chunk 1
    assert chunks[0].text.split()[-20:] == chunks[1].text.split()[:20]


def test_clean_text():
    assert clean_text("inte-\ngration   works") == "integration works"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
