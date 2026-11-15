"""Main command (exact form required by the assignment):

    python run_integrity_check.py --input_dir ./sample_submissions --output_dir ./reports
"""
import argparse
import logging
from pathlib import Path

import config
from src.ai_detector import ai_generated_probability
from src.boilerplate import find_boilerplate
from src.chunking import build_units, chunk_document
from src.extractors import extract_file
from src.external_match import find_external_matches, load_reference_index
from src.indexer import build_index, embed_chunks
from src.peer_similarity import find_peer_matches, peer_percentages
from src.reporter import (build_report, build_skipped_report, pairwise_matrix,
                          write_json, write_markdown, write_summary)

# Root level WARNING hides the noisy HTTP/model-download messages from other libraries;
# our own messages use the "integrity" logger, which is set to INFO.
logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger("integrity")
logger.setLevel(logging.INFO)


def main():
    parser = argparse.ArgumentParser(description="Multimodal submission integrity check")
    parser.add_argument("--input_dir", default=config.DEFAULT_INPUT_DIR)
    parser.add_argument("--output_dir", default=config.DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    input_dir, output_dir = Path(args.input_dir), Path(args.output_dir)
    if not input_dir.is_dir():
        logger.error("Input folder not found: %s", input_dir)
        return
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Extract and chunk every file (a bad file is skipped, never a crash)
    official, windows, skipped = {}, [], {}
    for path in sorted(input_dir.iterdir()):
        if not path.is_file() or path.name == "dataset_manifest.json":
            continue
        result = extract_file(path)
        if result.status != "ok":
            skipped[result.doc_id] = result.warning
            continue
        official[result.doc_id] = chunk_document(result)   # ~180 words, for AI detection
        windows.extend(build_units(result))                # small units, for matching
    logger.info("Readable files: %d | Skipped: %d", len(official), len(skipped))

    # 2. Embed, index, and compute the matches
    boilerplate, peer, external, peer_pct, ext_pct = set(), {}, {}, {}, {}
    if windows:
        vectors = embed_chunks(windows)
        index = build_index(vectors)
        boilerplate = find_boilerplate(windows, vectors, index)
        peer = find_peer_matches(windows, vectors, index, boilerplate)
        ref_chunks, ref_index = load_reference_index()
        external = find_external_matches(windows, vectors, ref_chunks, ref_index, boilerplate)
        peer_pct = peer_percentages(windows, boilerplate, peer)
        ext_pct = peer_percentages(windows, boilerplate, external)

    # 3. One JSON + one Markdown report per file, then the batch summary
    reports = []
    for doc_id in official:
        n_template = sum(1 for i, c in enumerate(windows)
                         if i in boilerplate and c.doc_id == doc_id)
        report = build_report(doc_id, peer_pct.get(doc_id, 0.0), ext_pct.get(doc_id, 0.0),
                              ai_generated_probability(official[doc_id]),
                              peer.get(doc_id, []), external.get(doc_id, []), n_template)
        reports.append(report)
        write_json(report, output_dir)
        write_markdown(report, output_dir)
        logger.info("%s -> risk %s", doc_id, report["total_integrity_risk"])
    for doc_id, warning in skipped.items():
        report = build_skipped_report(doc_id, warning)
        reports.append(report)
        write_json(report, output_dir)
        write_markdown(report, output_dir)
        logger.info("%s -> NOT_ASSESSED (%s)", doc_id, warning)

    write_summary(reports, pairwise_matrix(windows, boilerplate, peer), output_dir)
    logger.info("Reports written to %s", output_dir.resolve())


if __name__ == "__main__":
    main()