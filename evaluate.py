"""Evaluate the pipeline against the ground truth in sample_submissions/dataset_manifest.json.

Run the main pipeline first (so the reports exist), then:

    python evaluate.py

Part A: which pairs of files look similar, and which threshold separates real matches.
Part B: how the final reports (risk level and the three scores) compare with the truth.
The results are also saved to reports/evaluation_results.txt.
"""
import argparse
import itertools
import json
import logging
from pathlib import Path

import numpy as np

import config
from src.boilerplate import find_boilerplate
from src.chunking import build_units
from src.extractors import extract_file
from src.indexer import build_index, embed_chunks

logging.basicConfig(level=logging.ERROR)  # hide per-file warnings; this script is about metrics
OUT = []


def emit(text=""):
    print(text)
    OUT.append(text)


def rates(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    if precision is None or recall is None:
        f1 = None
    else:
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def fmt(x):
    return "n/a" if x is None else f"{x:.2f}"


def show_confusion(title, tp, fp, fn, tn):
    precision, recall, f1 = rates(tp, fp, fn)
    emit(f"\n{title}")
    emit("                    predicted YES   predicted NO")
    emit(f"  actual YES        {tp:>9}       {fn:>9}")
    emit(f"  actual NO         {fp:>9}       {tn:>9}")
    emit(f"  precision {fmt(precision)} | recall {fmt(recall)} | F1 {fmt(f1)}")


# ------------------------------------------------------------------ Part A
def part_a(input_dir: Path, manifest: dict):
    windows = []
    for path in sorted(input_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in (".pdf", ".docx", ".mp4"):
            result = extract_file(path)
            if result.status == "ok":
                windows.extend(build_units(result))
    vectors = embed_chunks(windows)
    boilerplate = find_boilerplate(windows, vectors, build_index(vectors))

    by_doc = {}
    for i, chunk in enumerate(windows):
        if i not in boilerplate:
            by_doc.setdefault(chunk.doc_id, []).append(i)
    sims = vectors @ vectors.T
    scores = {(a, b): float(sims[np.ix_(by_doc[a], by_doc[b])].max())
              for a, b in itertools.combinations(sorted(by_doc), 2)}

    true_pairs = {frozenset(p) for p in manifest["peer_match_pairs"]}
    group = set(manifest["benign_template_group"])
    is_true = lambda pair: frozenset(pair) in true_pairs

    emit("=" * 70)
    emit("PART A: pair similarity (best window-to-window cosine score per pair of files)")
    emit("=" * 70)
    emit(f"{len(by_doc)} readable files, {len(scores)} pairs, {len(true_pairs)} real matches, "
         f"{sum(1 for i in range(len(windows)) if i in boilerplate)} template windows ignored\n")
    emit(f"{'score':>6}  {'truth':5}  pair")
    for pair, s in sorted(scores.items(), key=lambda kv: -kv[1])[:16]:
        note = "  (same-case answers)" if pair[0] in group and pair[1] in group else ""
        emit(f"{s:6.2f}  {'MATCH' if is_true(pair) else 'no':5}  {pair[0]} <-> {pair[1]}{note}")

    pos = {p: s for p, s in scores.items() if is_true(p)}
    neg = {p: s for p, s in scores.items() if not is_true(p)}
    if pos:
        low = min(pos, key=pos.get)
        emit(f"\nLowest score among REAL matches : {pos[low]:.2f}  ({low[0]} <-> {low[1]})")
    if neg:
        high = max(neg, key=neg.get)
        emit(f"Highest score among NON-matches : {neg[high]:.2f}  ({high[0]} <-> {high[1]})")

    emit("\nThreshold sweep (a pair is predicted 'match' if its score >= threshold):")
    emit(f"{'threshold':>9}  {'TP':>3} {'FP':>3} {'FN':>3} {'TN':>4}  {'precision':>9} {'recall':>6} {'F1':>5}")
    for t in np.arange(0.50, 0.951, 0.05):
        tp = sum(1 for p, s in scores.items() if is_true(p) and s >= t)
        fp = sum(1 for p, s in scores.items() if not is_true(p) and s >= t)
        fn = sum(1 for p, s in scores.items() if is_true(p) and s < t)
        tn = sum(1 for p, s in scores.items() if not is_true(p) and s < t)
        precision, recall, f1 = rates(tp, fp, fn)
        mark = "  <- current" if abs(t - config.SIMILARITY_THRESHOLD) < 1e-9 else ""
        emit(f"{t:9.2f}  {tp:>3} {fp:>3} {fn:>3} {tn:>4}  {fmt(precision):>9} {fmt(recall):>6} "
             f"{fmt(f1):>5}{mark}")


# ------------------------------------------------------------------ Part B
def part_b(reports_dir: Path, manifest: dict):
    reports = {}
    for path in reports_dir.glob("*.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        if "document_id" in data:
            reports[data["document_id"]] = data

    emit("\n" + "=" * 70)
    emit("PART B: final reports versus ground truth")
    emit("=" * 70)
    emit(f"{'file':24} {'expected':9} {'got':13} {'peer%':>6} {'ext%':>6} {'AI%':>6}")

    peer_true = {d for pair in manifest["peer_match_pairs"] for d in pair}
    ext_true = set(manifest["external_matches"])
    ai_true = set(manifest["ai_generated"])
    human_true = set(manifest.get("human_written", []))
    flag = dict(tp=0, fp=0, fn=0, tn=0)
    peer = dict(tp=0, fp=0, fn=0, tn=0)
    ext = dict(tp=0, fp=0, fn=0, tn=0)
    ai = dict(tp=0, fp=0, fn=0, tn=0)
    skip_ok = skip_total = 0
    wrong = []

    def count(table, truth, predicted):
        key = ("tp" if truth else "fp") if predicted else ("fn" if truth else "tn")
        table[key] += 1

    for item in manifest["files"]:
        name, expected = item["file"], item["expected_flagged"]
        report = reports.get(name)
        if report is None:
            emit(f"{name:24} {'?':9} {'NO REPORT':13}")
            continue
        risk, s = report["total_integrity_risk"], report["scores"]
        if expected is None:
            skip_total += 1
            ok = risk == "NOT_ASSESSED"
            skip_ok += ok
            emit(f"{name:24} {'skipped':9} {risk:13} {'-':>6} {'-':>6} {'-':>6}  {'ok' if ok else 'WRONG'}")
            continue
        peer_pct = s["peer_similarity_percentage"] or 0.0
        ext_pct = s["external_similarity_percentage"] or 0.0
        ai_pct = s["ai_generated_probability"] or 0.0
        flagged = risk in ("MEDIUM", "HIGH")
        count(flag, expected, flagged)
        count(peer, name in peer_true, peer_pct >= config.RISK_MEDIUM["peer"])
        count(ext, name in ext_true, ext_pct >= config.RISK_MEDIUM["external"])
        if name in ai_true or name in human_true:   # only files with known authorship
            count(ai, name in ai_true, ai_pct >= config.RISK_MEDIUM["ai"])
        ok = flagged == expected
        if not ok:
            wrong.append((name, expected, risk, peer_pct, ext_pct, ai_pct))
        emit(f"{name:24} {'flagged' if expected else 'clean':9} {risk:13} {peer_pct:6.1f} "
             f"{ext_pct:6.1f} {ai_pct:6.1f}  {'ok' if ok else 'WRONG'}")

    emit(f"\nUnreadable files correctly skipped: {skip_ok} of {skip_total}")
    show_confusion("Overall (flagged = MEDIUM or HIGH risk)", **flag)
    show_confusion(f"Peer signal (peer % >= {config.RISK_MEDIUM['peer']:.0f})", **peer)
    show_confusion(f"External signal (external % >= {config.RISK_MEDIUM['external']:.0f})", **ext)
    show_confusion(f"AI signal (AI % >= {config.RISK_MEDIUM['ai']:.0f}; only files with known authorship)", **ai)

    if wrong:
        emit("\nFiles the system got wrong:")
        for name, expected, risk, p, e, a in wrong:
            kind = "false alarm" if not expected else "missed"
            emit(f"  {name}: {kind} (risk {risk}; peer {p:.0f}%, external {e:.0f}%, AI {a:.0f}%)")


def main():
    parser = argparse.ArgumentParser(description="Evaluate against dataset_manifest.json")
    parser.add_argument("--input_dir", default=config.DEFAULT_INPUT_DIR)
    parser.add_argument("--reports_dir", default=config.DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    input_dir, reports_dir = Path(args.input_dir), Path(args.reports_dir)

    manifest = json.loads((input_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
    part_a(input_dir, manifest)
    part_b(reports_dir, manifest)
    (reports_dir / "evaluation_results.txt").write_text("\n".join(OUT) + "\n", encoding="utf-8")
    print(f"\nSaved to {reports_dir / 'evaluation_results.txt'}")


if __name__ == "__main__":
    main()