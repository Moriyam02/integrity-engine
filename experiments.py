
import argparse
import itertools
import json
import logging
from pathlib import Path

import numpy as np

import config
from evaluate import rates
from src.boilerplate import find_boilerplate
from src.chunking import chunk_by_sentences, chunk_document
from src.extractors import extract_file
from src.indexer import build_index

logging.basicConfig(level=logging.ERROR)

DEFAULT_MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",         # current choice
    "sentence-transformers/paraphrase-MiniLM-L6-v2",  # same size, trained on paraphrases
    "sentence-transformers/all-mpnet-base-v2",        # larger and slower
]
# (label, kind, size, overlap): words = sliding word windows, sentences = whole sentences
UNIT_CONFIGS = [
    ("words 40/10", "words", 40, 10),
    ("words 80/20", "words", 80, 20),
    ("sentences x1", "sentences", 1, None),
    ("sentences x2", "sentences", 2, None),
    ("sentences x3", "sentences", 3, None),
]
SIM_GRID = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
LINES = []


def emit(text=""):
    print(text)
    LINES.append(text)


def make_units(result, kind, size, overlap):
    if kind == "words":
        return chunk_document(result, size, overlap)
    return chunk_by_sentences(result, size)


def covers(best_scores, t, coverage_min):
    """True if enough of a file's units have a partner at or above t."""
    matched = int((best_scores >= t).sum())
    return (matched / len(best_scores) >= coverage_min
            and matched >= min(config.MIN_MATCHED_UNITS, len(best_scores)))


def score_setting(units, vectors, true_pairs, coverage_min):
    """Return (lowest real-match score, highest non-match score, best (t, tp, fp, fn, f1))."""
    boilerplate = find_boilerplate(units, vectors, build_index(vectors))
    by_doc = {}
    for i, unit in enumerate(units):
        if i not in boilerplate:
            by_doc.setdefault(unit.doc_id, []).append(i)
    sims = vectors @ vectors.T
    pairs = []
    for a, b in itertools.combinations(sorted(by_doc), 2):
        sub = sims[np.ix_(by_doc[a], by_doc[b])]
        pairs.append((frozenset((a, b)) in true_pairs, float(sub.max()),
                      sub.max(axis=1), sub.max(axis=0)))
    real = [m for truth, m, _, _ in pairs if truth]
    other = [m for truth, m, _, _ in pairs if not truth]

    best = None
    for t in SIM_GRID:
        tp = fp = fn = 0
        for truth, _, rows, cols in pairs:
            predicted = covers(rows, t, coverage_min) or covers(cols, t, coverage_min)
            if truth and predicted:
                tp += 1
            elif predicted:
                fp += 1
            elif truth:
                fn += 1
        f1 = rates(tp, fp, fn)[2] or 0.0
        if best is None or (f1, -fp, t) > (best[4], -best[2], best[0]):
            best = (t, tp, fp, fn, f1)
    return (min(real) if real else float("nan"), max(other) if other else float("nan"), best)


def main():
    parser = argparse.ArgumentParser(description="Compare unit types and embedding models")
    parser.add_argument("--input_dir", default=config.DEFAULT_INPUT_DIR)
    parser.add_argument("--reports_dir", default=config.DEFAULT_OUTPUT_DIR)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    args = parser.parse_args()
    input_dir, reports_dir = Path(args.input_dir), Path(args.reports_dir)

    from sentence_transformers import SentenceTransformer

    manifest = json.loads((input_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
    true_pairs = {frozenset(p) for p in manifest["peer_match_pairs"]}
    coverage_min = config.RISK_MEDIUM["peer"] / 100.0
    results = []
    for path in sorted(input_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in (".pdf", ".docx", ".mp4"):
            result = extract_file(path)
            if result.status == "ok":
                results.append(result)

    emit(f"{len(results)} readable files, {len(true_pairs)} real matching pairs.")
    emit(f"A pair counts as 'match' when at least {coverage_min:.0%} of the units of one file "
         f"have a partner at or above t in the other file (and at least "
         f"{config.MIN_MATCHED_UNITS} units match).")
    emit("real-min = lowest best-score among real matches; benign-max = highest among the rest.\n")
    emit(f"{'model':30} {'units':13} {'real-min':>8} {'benign-max':>10} | "
         f"{'best t':>6} {'TP':>3} {'FP':>3} {'FN':>3} {'F1':>5}")
    for name in args.models:
        model = SentenceTransformer(name)
        for label, kind, size, overlap in UNIT_CONFIGS:
            units = [u for r in results for u in make_units(r, kind, size, overlap)]
            vectors = np.asarray(model.encode([u.text for u in units], normalize_embeddings=True,
                                              show_progress_bar=False), dtype="float32")
            low, high, (t, tp, fp, fn, f1) = score_setting(units, vectors, true_pairs, coverage_min)
            emit(f"{name.split('/')[-1]:30} {label:13} {low:8.2f} {high:10.2f} | "
                 f"{t:6.2f} {tp:>3} {fp:>3} {fn:>3} {f1:5.2f}")
        emit("")

    (reports_dir / "experiment_results.txt").write_text("\n".join(LINES) + "\n", encoding="utf-8")
    print(f"Saved to {reports_dir / 'experiment_results.txt'}")


if __name__ == "__main__":
    main()