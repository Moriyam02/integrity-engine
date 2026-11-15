"""Task 2a: turn chunks into embeddings and store them in a FAISS index."""
from typing import List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

import config
from src.chunking import Chunk

_model = None  # loaded once, then reused


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(config.EMBEDDING_MODEL)
    return _model


def embed_chunks(chunks: List[Chunk]) -> np.ndarray:
    """One 384-number vector per chunk. Vectors are normalised to length 1,
    so the inner product of two vectors equals their cosine similarity."""
    texts = [c.text for c in chunks]
    vectors = get_model().encode(texts, normalize_embeddings=True,
                                 show_progress_bar=False)
    return np.asarray(vectors, dtype="float32")


def build_index(vectors: np.ndarray) -> faiss.IndexFlatIP:
    """Exact search index. IP = inner product (= cosine here)."""
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index
