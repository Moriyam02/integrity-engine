"""Task 1b: clean text and cut it into overlapping chunks.

Idea: turn the whole document into one list of words, where each word
remembers its page or timestamp. Then slide a window over that list.
Each chunk can then say exactly which page / time range it came from.
"""
import re
import unicodedata
from dataclasses import dataclass
from typing import List, Optional

import config
from src.extractors import ExtractionResult


@dataclass
class Chunk:
    doc_id: str
    chunk_id: str                      # e.g. "report_01.pdf#3"
    text: str
    file_type: str
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    time_start: Optional[float] = None
    time_end: Optional[float] = None


def clean_text(text: str) -> str:
    """Fix odd characters, hyphenated line breaks and extra whitespace."""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"-\s*\n\s*", "", text)   # "inte-\ngration" -> "integration"
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def format_timestamp(seconds: Optional[float]) -> str:
    """12.5 -> '00:00:12' (the format used in the report)."""
    if seconds is None:
        return ""
    s = int(seconds)
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def chunk_document(result: ExtractionResult,
                   chunk_words: int = config.CHUNK_WORDS,
                   overlap: int = config.CHUNK_OVERLAP_WORDS) -> List[Chunk]:
    if result.status != "ok":
        return []

    # 1. one flat list of (word, segment) pairs
    words, owners = [], []
    for seg in result.segments:
        for w in clean_text(seg.text).split():
            words.append(w)
            owners.append(seg)
    if not words:
        return []

    # 2. slide a window over the words
    step = chunk_words - overlap
    chunks, start = [], 0
    while True:
        end = min(start + chunk_words, len(words))
        segs = owners[start:end]
        pages = [s.page for s in segs if s.page is not None]
        starts = [s.start for s in segs if s.start is not None]
        ends = [s.end for s in segs if s.end is not None]
        chunks.append(Chunk(
            doc_id=result.doc_id,
            chunk_id=f"{result.doc_id}#{len(chunks)}",
            text=" ".join(words[start:end]),
            file_type=result.file_type,
            page_start=min(pages) if pages else None,
            page_end=max(pages) if pages else None,
            time_start=min(starts) if starts else None,
            time_end=max(ends) if ends else None,
        ))
        if end >= len(words):
            break
        start += step
    return chunks


def location_label(chunk: Chunk) -> str:
    """Human-readable place a chunk came from: a page range or a video time range."""
    if chunk.time_start is not None:
        return f"{format_timestamp(chunk.time_start)} - {format_timestamp(chunk.time_end)}"
    if chunk.page_start is not None:
        if chunk.page_start == chunk.page_end:
            return f"page {chunk.page_start}"
        return f"pages {chunk.page_start}-{chunk.page_end}"
    return "n/a (no page or time info)"


# ------------------------------------------------- sentence-based units
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def chunk_by_sentences(result: ExtractionResult, per_unit: int = 1,
                       min_words: int = 6, max_words: int = 60) -> List[Chunk]:
    """Cut a document into units of `per_unit` whole sentences (no overlap).

    Sentences line up better between a text and its paraphrase than fixed word
    windows do. Very short sentences are glued to a neighbour, and very long
    ones (no punctuation) are cut into pieces, so every unit has a sensible size.
    Each unit keeps the page / timestamp of the text it came from.
    """
    if result.status != "ok":
        return []

    # 1. every sentence, with the segment (page / time) it came from
    sentences = []
    for seg in result.segments:
        for sent in _SENTENCE_END.split(clean_text(seg.text)):
            words = sent.split()
            for i in range(0, len(words), max_words):
                sentences.append([" ".join(words[i:i + max_words]), [seg]])

    # 2. glue too-short sentences to the next one (or the previous one at the end)
    merged, pending = [], None
    for text, segs in sentences:
        if pending:
            text, segs = pending[0] + " " + text, pending[1] + segs
            pending = None
        if len(text.split()) < min_words:
            pending = [text, segs]
        else:
            merged.append([text, segs])
    if pending:
        if merged:
            merged[-1][0] += " " + pending[0]
            merged[-1][1] += pending[1]
        else:
            merged.append(pending)

    # 3. group `per_unit` sentences into one unit
    chunks = []
    for start in range(0, len(merged), per_unit):
        group = merged[start:start + per_unit]
        segs = [s for _, group_segs in group for s in group_segs]
        pages = [s.page for s in segs if s.page is not None]
        starts = [s.start for s in segs if s.start is not None]
        ends = [s.end for s in segs if s.end is not None]
        chunks.append(Chunk(
            doc_id=result.doc_id,
            chunk_id=f"{result.doc_id}#{len(chunks)}",
            text=" ".join(text for text, _ in group),
            file_type=result.file_type,
            page_start=min(pages) if pages else None,
            page_end=max(pages) if pages else None,
            time_start=min(starts) if starts else None,
            time_end=max(ends) if ends else None,
        ))
    return chunks


def build_units(result: ExtractionResult) -> List[Chunk]:
    """The small units used for matching, as selected in config.py."""
    if config.MATCH_UNIT == "sentences":
        return chunk_by_sentences(result, config.MATCH_SENTENCES)
    return chunk_document(result, config.MATCH_WINDOW_WORDS, config.MATCH_WINDOW_OVERLAP)