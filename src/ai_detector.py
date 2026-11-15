"""Task 3b: estimate how likely a document is AI-generated.

We use an open-source Hugging Face RoBERTa classifier. Each official chunk
(about 180 words) is scored separately, and the document score is the average.
If the model cannot run, or the text is too short, we return None instead of
guessing, so the report can say "not available" honestly.
"""
import logging
from typing import List, Optional

import config
from src.chunking import Chunk

logger = logging.getLogger(__name__)
_classifier = None  # loaded once, then reused


def _get_classifier():
    global _classifier
    if _classifier is None:
        from transformers import pipeline
        _classifier = pipeline("text-classification", model=config.AI_DETECTOR_MODEL,
                               truncation=True, max_length=512)
    return _classifier


def ai_generated_probability(chunks: List[Chunk]) -> Optional[float]:
    """Average chance (0-100) that the chunks are AI-written, or None if unknown."""
    texts = [c.text for c in chunks if len(c.text.split()) >= config.AI_MIN_WORDS]
    if not texts:
        logger.warning("Text too short for AI detection")
        return None
    try:
        results = _get_classifier()(texts, top_k=None, batch_size=8)
        probs = []
        for scores in results:  # scores = [{"label": ..., "score": ...}, ...]
            ai = next(s["score"] for s in scores
                      if s["label"].lower() == config.AI_LABEL.lower())
            probs.append(ai)
    except Exception as err:
        logger.warning("AI detector unavailable (%s: %s)", type(err).__name__, err)
        return None
    return 100.0 * sum(probs) / len(probs)