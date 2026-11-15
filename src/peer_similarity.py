"""Task 2: compare every document with all the OTHER documents.

For each window of a document we look for its single best match in a
different document. The peer similarity percentage of a document is the
share of its windows that have such a match.
"""
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Set

import numpy as np

import config
from src.chunking import Chunk


@dataclass
class PeerMatch:
    query: Chunk      # the passage in the document being checked
    matched: Chunk    # the passage it matches in another document
    score: float      # cosine similarity, 0 to 1


def find_peer_matches(chunks: List[Chunk], vectors: np.ndarray, index,
                      boilerplate: Set[int]) -> Dict[str, List[PeerMatch]]:
    """Best cross-document match per window. Template windows are ignored."""
    scores, ids = index.search(vectors, len(chunks))
    matches = {c.doc_id: [] for c in chunks}
    for i, chunk in enumerate(chunks):
        if i in boilerplate:
            continue
        for score, j in zip(scores[i], ids[i]):
            if score < config.SIMILARITY_THRESHOLD:
                break  # sorted high to low, so nothing later can qualify
            if j in boilerplate or chunks[j].doc_id == chunk.doc_id:
                continue  # self-match exclusion: same document or template
            matches[chunk.doc_id].append(PeerMatch(chunk, chunks[j], float(score)))
            break
    totals = Counter(c.doc_id for i, c in enumerate(chunks) if i not in boilerplate)
    for doc, found in matches.items():
        if len(found) < min(config.MIN_MATCHED_UNITS, totals[doc]):
            matches[doc] = []   # a lone matching unit is a coincidence, not evidence
    return matches


def peer_percentages(chunks: List[Chunk], boilerplate: Set[int],
                     matches: Dict[str, List[PeerMatch]]) -> Dict[str, float]:
    """Percent of each document's (non-template) windows that have a peer match."""
    totals = Counter(c.doc_id for i, c in enumerate(chunks) if i not in boilerplate)
    return {doc: (100.0 * len(found) / totals[doc] if totals[doc] else 0.0)
            for doc, found in matches.items()}