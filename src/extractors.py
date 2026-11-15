"""Task 1a: pull text out of PDF, DOCX and MP4 files.

Every extractor returns an ExtractionResult. A bad file never raises an
error: it comes back with status="skipped" and a human-readable warning.
"""
import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import config

logger = logging.getLogger(__name__)


@dataclass
class Segment:
    """A piece of text plus where it came from."""
    text: str
    page: Optional[int] = None     # PDF page number (1-based)
    start: Optional[float] = None  # video start time in seconds
    end: Optional[float] = None    # video end time in seconds


@dataclass
class ExtractionResult:
    doc_id: str                    # file name, e.g. "report_01.pdf"
    file_type: str                 # "pdf", "docx", "mp4"
    status: str = "ok"             # "ok" or "skipped"
    warning: str = ""
    segments: List[Segment] = field(default_factory=list)


def _skipped(path: Path, file_type: str, reason: str) -> ExtractionResult:
    logger.warning("Skipping %s: %s", path.name, reason)
    return ExtractionResult(path.name, file_type, "skipped", reason)


# ---------------------------------------------------------------- PDF
def _normalise_line(line: str) -> str:
    """Make 'Page 3' and 'Page 4' look the same so repeats can be counted."""
    return re.sub(r"\d+", "#", line.strip().lower())


def _remove_headers_footers(pages: List[List[str]]) -> List[List[str]]:
    """Drop lines that repeat on most pages, and bare page numbers."""
    pages = [[l for l in page if not re.fullmatch(r"\s*(page\s*)?\d+(\s*(of|/)\s*\d+)?\s*", l, re.I)]
             for page in pages]
    if len(pages) < config.MIN_PAGES_FOR_HEADER_DETECTION:
        return pages
    counts = Counter()
    for page in pages:
        for line in set(_normalise_line(l) for l in page if l.strip()):
            counts[line] += 1
    limit = config.REPEATED_LINE_RATIO * len(pages)
    repeated = {line for line, c in counts.items() if c >= limit}
    return [[l for l in page if _normalise_line(l) not in repeated] for page in pages]


def extract_pdf(path: Path) -> ExtractionResult:
    from pypdf import PdfReader
    if path.stat().st_size == 0:
        return _skipped(path, "pdf", "file is empty (0 bytes)")
    try:
        reader = PdfReader(str(path))
        if reader.is_encrypted:
            reader.decrypt("")
        pages = [(p.extract_text() or "").splitlines() for p in reader.pages]
    except Exception as err:
        return _skipped(path, "pdf", f"corrupted or unreadable PDF ({type(err).__name__}: {err})")

    pages = _remove_headers_footers(pages)
    segments = [Segment(text="\n".join(lines), page=i + 1)
                for i, lines in enumerate(pages) if "".join(lines).strip()]
    if sum(len(s.text) for s in segments) < 20:
        return _skipped(path, "pdf", "no extractable text (maybe a scanned image PDF)")
    return ExtractionResult(path.name, "pdf", segments=segments)


# --------------------------------------------------------------- DOCX
def _paragraph_pages(paragraphs) -> List[int]:
    """Estimated page number of every paragraph in a DOCX file.

    Word files do not store fixed pages. We use the page-break markers that Word
    saved in the file when there are any; otherwise we count about
    config.DOCX_WORDS_PER_PAGE words per page.
    """
    xml = [p._p.xml for p in paragraphs]
    marker = "lastRenderedPageBreak"
    if not any(marker in x for x in xml):
        marker = 'w:type="page"'          # an explicit page break
    has_markers = any(marker in x for x in xml)

    pages, page, words_on_page = [], 1, 0
    for paragraph, x in zip(paragraphs, xml):
        if has_markers:
            page += x.count(marker)
        else:
            n_words = len(paragraph.text.split())
            if words_on_page and words_on_page + n_words > config.DOCX_WORDS_PER_PAGE:
                page += 1
                words_on_page = 0
            words_on_page += n_words
        pages.append(page)
    return pages


def extract_docx(path: Path) -> ExtractionResult:
    from docx import Document
    if path.stat().st_size == 0:
        return _skipped(path, "docx", "file is empty (0 bytes)")
    try:
        doc = Document(str(path))
        pages = _paragraph_pages(doc.paragraphs)
        segments = [Segment(text=p.text, page=page)
                    for p, page in zip(doc.paragraphs, pages) if p.text.strip()]
        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    if cell.text.strip():
                        segments.append(Segment(text=cell.text, page=pages[-1] if pages else 1))
    except Exception as err:
        return _skipped(path, "docx", f"corrupted or unreadable DOCX ({type(err).__name__}: {err})")

    if sum(len(s.text) for s in segments) < 20:
        return _skipped(path, "docx", "document has no readable text")
    return ExtractionResult(path.name, "docx", segments=segments)


# ---------------------------------------------------------------- MP4
_whisper_model = None  # loaded once, then reused


def _get_whisper():
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel
        _whisper_model = WhisperModel(config.WHISPER_MODEL, device="cpu", compute_type="int8")
    return _whisper_model


def _has_audio_track(path: Path) -> bool:
    """True if the video file contains at least one audio stream."""
    import av
    with av.open(str(path)) as container:
        return len(container.streams.audio) > 0


def extract_mp4(path: Path) -> ExtractionResult:
    if path.stat().st_size == 0:
        return _skipped(path, "mp4", "file is empty (0 bytes)")
    try:
        if not _has_audio_track(path):
            return _skipped(path, "mp4", "video has no audio track (nothing to transcribe)")
        model = _get_whisper()
        # vad_filter skips silence so a silent video gives no fake text
        whisper_segments, _info = model.transcribe(str(path), language="en", vad_filter=True)
        segments = [Segment(text=s.text.strip(), start=s.start, end=s.end)
                    for s in whisper_segments if s.text.strip()]
    except Exception as err:
        return _skipped(path, "mp4", f"cannot read audio, maybe no audio track ({type(err).__name__}: {err})")

    if sum(len(s.text.split()) for s in segments) < config.MIN_TRANSCRIPT_WORDS:
        return _skipped(path, "mp4", "no speech detected (silent video)")
    return ExtractionResult(path.name, "mp4", segments=segments)


# --------------------------------------------------------- dispatcher
EXTRACTORS = {".pdf": extract_pdf, ".docx": extract_docx, ".mp4": extract_mp4}


def extract_file(path: Path) -> ExtractionResult:
    """Pick the right extractor. Never raises."""
    suffix = path.suffix.lower()
    func = EXTRACTORS.get(suffix)
    if func is None:
        return _skipped(path, suffix.lstrip("."), f"unsupported file type '{suffix}'")
    try:
        return func(path)
    except Exception as err:  # last safety net
        return _skipped(path, suffix.lstrip("."), f"unexpected error ({type(err).__name__}: {err})")