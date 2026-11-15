"""Download the open-source models once, so the pipeline can run offline.

Usage:  python download_models.py
"""
import config


def main():
    print(f"1/3 Whisper '{config.WHISPER_MODEL}' (speech to text)...")
    from faster_whisper import WhisperModel
    WhisperModel(config.WHISPER_MODEL, device="cpu", compute_type="int8")

    print(f"2/3 Embedding model {config.EMBEDDING_MODEL}...")
    from sentence_transformers import SentenceTransformer
    SentenceTransformer(config.EMBEDDING_MODEL)

    print(f"3/3 AI-text detector {config.AI_DETECTOR_MODEL}...")
    from transformers import pipeline
    pipeline("text-classification", model=config.AI_DETECTOR_MODEL)

    print("Done. All models are cached locally.")


if __name__ == "__main__":
    main()