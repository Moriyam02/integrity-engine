"""Task 3a: external similarity against a local reference knowledge base.

Full web search costs money and hits rate limits, so (as the assignment
allows) we compare submissions against a folder of reference .txt files,
for example Wikipedia articles or the original case-study sources.
"""
import logging
from collections import Counter
from pathlib import Path
from typing import Dict, List, Set

import numpy as np

import config
from src.chunking import Chunk, build_units
from src.extractors import ExtractionResult, Segment
from src.indexer import build_index, embed_chunks
from src.peer_similarity import PeerMatch

logger = logging.getLogger(__name__)


def load_reference_index(folder: str = config.REFERENCE_DIR):
    """Read every .txt file in the folder, cut it into units and index it.
    Returns (reference_chunks, index), or ([], None) if there is nothing to use."""
    folder = Path(folder)
    ref_chunks: List[Chunk] = []
    if folder.is_dir():
        for path in sorted(folder.glob("*.txt")):
            text = path.read_text(encoding="utf-8", errors="ignore")
            result = ExtractionResult(path.name, "txt", segments=[Segment(text=text)])
            ref_chunks.extend(build_units(result))
    if not ref_chunks:
        logger.warning("No reference texts found in %s: external match will be 0%%", folder)
        return [], None
    return ref_chunks, build_index(embed_chunks(ref_chunks))


def find_external_matches(chunks: List[Chunk], vectors: np.ndarray,
                          ref_chunks: List[Chunk], ref_index,
                          boilerplate: Set[int]) -> Dict[str, List[PeerMatch]]:
    """Best reference-text match for each submission window (templates ignored)."""
    matches = {c.doc_id: [] for c in chunks}
    if ref_index is None:
        return matches
    scores, ids = ref_index.search(vectors, 1)   # best single match per window
    for i, chunk in enumerate(chunks):
        if i in boilerplate:
            continue
        score, j = float(scores[i][0]), int(ids[i][0])
        if score >= config.EXTERNAL_SIMILARITY_THRESHOLD:
            matches[chunk.doc_id].append(PeerMatch(chunk, ref_chunks[j], score))
    totals = Counter(c.doc_id for i, c in enumerate(chunks) if i not in boilerplate)
    for doc, found in matches.items():
        if len(found) < min(config.MIN_MATCHED_UNITS, totals[doc]):
            matches[doc] = []   # a lone matching unit is a coincidence, not evidence
    return matches