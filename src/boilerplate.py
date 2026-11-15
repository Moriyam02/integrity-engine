"""Scenario 4: find text that is repeated across many documents (templates).

Idea: if almost the same passage shows up in 3 or more OTHER documents, it is
probably a shared case prompt, instruction block or citation list, not
copying between two students. Those passages are ignored when scoring.
A genuine copy between two documents appears in only 1 other document,
so it is NOT treated as boilerplate.
"""
from typing import List, Set

import numpy as np

import config
from src.chunking import Chunk


def find_boilerplate(chunks: List[Chunk], vectors: np.ndarray, index) -> Set[int]:
    """Return the positions (in `chunks`) of windows that are template text."""
    scores, ids = index.search(vectors, len(chunks))
    flagged = set()
    for i, chunk in enumerate(chunks):
        other_docs = set()
        for score, j in zip(scores[i], ids[i]):
            if score < config.BOILERPLATE_SIMILARITY:
                break  # results are sorted, so nothing later can be higher
            if chunks[j].doc_id != chunk.doc_id:
                other_docs.add(chunks[j].doc_id)
        if len(other_docs) >= config.BOILERPLATE_MIN_DOCS:
            flagged.add(i)
    return flagged