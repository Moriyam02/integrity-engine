"""Task 4: build the explainable report for each document and save it as JSON.

The JSON follows the structure given in the assignment. Extra keys
("external_matches", "warnings") are added without changing the required ones.
"""
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import config
from src.chunking import Chunk, format_timestamp
from src.peer_similarity import PeerMatch


def _where(chunk: Chunk):
    """Page number ('2-3' for a range) for documents, time range for videos."""
    if chunk.file_type == "mp4":
        return f"{format_timestamp(chunk.time_start)} - {format_timestamp(chunk.time_end)}"
    if chunk.page_start is None:
        return None
    if chunk.page_start == chunk.page_end:
        return chunk.page_start
    return f"{chunk.page_start}-{chunk.page_end}"


def _jaccard(a: str, b: str) -> float:
    """Word overlap: shared words / all words (0 = nothing shared, 1 = same words).
    Low Jaccard with a high similarity score means the text was reworded."""
    words_a = set(re.findall(r"[a-z0-9']+", a.lower()))
    words_b = set(re.findall(r"[a-z0-9']+", b.lower()))
    union = words_a | words_b
    return len(words_a & words_b) / len(union) if union else 0.0


def _match_entry(m: PeerMatch) -> dict:
    query_key = "query_timestamp" if m.query.file_type == "mp4" else "query_page"
    matched_key = "matched_timestamp" if m.matched.file_type == "mp4" else "matched_page"
    return {
        "matched_file": m.matched.doc_id,
        "similarity_score": round(m.score, 2),
        "word_overlap_jaccard": round(_jaccard(m.query.text, m.matched.text), 2),
        "query_passage": m.query.text,
        "matched_passage": m.matched.text,
        "location": {query_key: _where(m.query), matched_key: _where(m.matched)},
    }


def split_matches(matches: List[PeerMatch]):
    """Cross-modal matches (video <-> written document) go in 'multimodal_matches'."""
    peer, multimodal = [], []
    for m in sorted(matches, key=lambda x: -x.score):
        crosses = (m.query.file_type == "mp4") != (m.matched.file_type == "mp4")
        (multimodal if crosses else peer).append(_match_entry(m))
    return peer, multimodal


def external_entries(matches: List[PeerMatch]) -> List[dict]:
    return [_match_entry(m) for m in sorted(matches, key=lambda x: -x.score)]


def assess_risk(peer: float, external: float, ai: Optional[float]) -> str:
    ai = ai or 0.0
    scores = {"peer": peer, "external": external, "ai": ai}
    if any(scores[k] >= config.RISK_HIGH[k] for k in scores):
        return "HIGH"
    if any(scores[k] >= config.RISK_MEDIUM[k] for k in scores):
        return "MEDIUM"
    return "LOW"


def build_explanation(risk, peer_pct, ext_pct, ai_prob, peer_e, multi_e, ext_e, n_template):
    parts = [f"Overall integrity risk: {risk}."]
    internal = peer_e + multi_e
    if internal:
        files = ", ".join(sorted({e["matched_file"] for e in internal}))
        best = max(e["similarity_score"] for e in internal)
        parts.append(f"{peer_pct:.0f}% of this document's passages closely match other "
                     f"submissions ({files}); the strongest match scores {best:.2f}.")
        if multi_e:
            parts.append(f"{len(multi_e)} of these are cross-modal matches (video transcript "
                         "versus written document).")
    if ext_e:
        sources = ", ".join(sorted({e["matched_file"] for e in ext_e}))
        parts.append(f"{ext_pct:.0f}% of passages closely match external reference texts "
                     f"({sources}).")
    if ai_prob is None:
        parts.append("An AI-content estimate is not available (text too short or detector "
                     "unavailable).")
    else:
        parts.append(f"The AI-text detector rates the writing {ai_prob:.0f}% likely "
                     "AI-generated (a statistical estimate, not proof).")
    if n_template:
        parts.append(f"{n_template} passage(s) repeated across several submissions (likely a "
                     "shared prompt or template) were ignored.")
    if not internal and not ext_e:
        parts.append("No significant overlap with other submissions or reference texts was found.")
    return " ".join(parts)


def build_report(doc_id, peer_pct, ext_pct, ai_prob, peer_matches, external_matches,
                 n_template) -> dict:
    peer_e, multi_e = split_matches(peer_matches)
    ext_e = external_entries(external_matches)
    risk = assess_risk(peer_pct, ext_pct, ai_prob)
    return {
        "document_id": doc_id,
        "total_integrity_risk": risk,
        "scores": {
            "peer_similarity_percentage": round(peer_pct, 1),
            "external_similarity_percentage": round(ext_pct, 1),
            "ai_generated_probability": None if ai_prob is None else round(ai_prob, 1),
        },
        "peer_matches": peer_e,
        "multimodal_matches": multi_e,
        "external_matches": ext_e,
        "explanations": build_explanation(risk, peer_pct, ext_pct, ai_prob,
                                          peer_e, multi_e, ext_e, n_template),
        "warnings": [],
    }


def build_skipped_report(doc_id: str, warning: str) -> dict:
    """Report for a file that could not be read (empty, corrupted, silent video...)."""
    return {
        "document_id": doc_id,
        "total_integrity_risk": "NOT_ASSESSED",
        "scores": {"peer_similarity_percentage": None,
                   "external_similarity_percentage": None,
                   "ai_generated_probability": None},
        "peer_matches": [],
        "multimodal_matches": [],
        "external_matches": [],
        "explanations": f"This file could not be analysed and was skipped: {warning}.",
        "warnings": [warning],
    }


def write_json(report: dict, output_dir: Path) -> Path:
    path = Path(output_dir) / f"{report['document_id']}.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


# ---------------------------------------------------------------- Markdown
def _pct(value) -> str:
    return "n/a" if value is None else f"{value}%"


def render_markdown(report: dict) -> str:
    """Human-readable version of one report, designed to be checked in 30 seconds."""
    s = report["scores"]
    lines = [
        f"# Integrity report: {report['document_id']}", "",
        f"**Overall risk: {report['total_integrity_risk']}**", "",
        "| Signal | Score |", "|---|---|",
        f"| Internal peer similarity | {_pct(s['peer_similarity_percentage'])} |",
        f"| External match | {_pct(s['external_similarity_percentage'])} |",
        f"| AI-generated likelihood | {_pct(s['ai_generated_probability'])} |", "",
        "## Summary", "", report["explanations"], "",
    ]
    all_entries = report["peer_matches"] + report["multimodal_matches"] + report["external_matches"]
    if (report["document_id"].lower().endswith(".docx")
            or any(e["matched_file"].lower().endswith(".docx") for e in all_entries)):
        lines += ["*Page numbers for .docx files are estimates: Word files do not store fixed pages.*", ""]
    sections = [("Matches with other submissions", "peer_matches"),
                ("Cross-modal matches (video and document)", "multimodal_matches"),
                ("Matches with external reference texts", "external_matches")]
    for title, key in sections:
        entries = report[key]
        if not entries:
            continue
        lines += [f"## {title}", ""]
        for n, e in enumerate(entries[:config.MD_MAX_MATCHES], 1):
            where = "; ".join(f"{k.replace('_', ' ')}: {'n/a' if v is None else v}"
                              for k, v in e["location"].items())
            lines += [f"### {n}. `{e['matched_file']}` (similarity {e['similarity_score']:.2f})", "",
                      f"*{where}; word overlap (Jaccard): {e['word_overlap_jaccard']:.2f}*", "",
                      f"> **This document:** {e['query_passage']}", ">",
                      f"> **Matched passage:** {e['matched_passage']}", ""]
        hidden = len(entries) - config.MD_MAX_MATCHES
        if hidden > 0:
            lines += [f"*{hidden} more matches are listed in the JSON report.*", ""]
    if report["warnings"]:
        lines += ["## Warnings", ""] + [f"- {w}" for w in report["warnings"]] + [""]
    return "\n".join(lines)


def write_markdown(report: dict, output_dir: Path) -> Path:
    path = Path(output_dir) / f"{report['document_id']}.md"
    path.write_text(render_markdown(report), encoding="utf-8")
    return path


# ----------------------------------------------------------- summary matrix
def pairwise_matrix(windows: List[Chunk], boilerplate, peer) -> Dict[tuple, float]:
    """(doc A, doc B) -> % of A's non-template windows whose best match is in B."""
    totals = Counter(c.doc_id for i, c in enumerate(windows) if i not in boilerplate)
    counts = Counter()
    for doc, matches in peer.items():
        for m in matches:
            counts[(doc, m.matched.doc_id)] += 1
    return {(a, b): 100.0 * n / totals[a] for (a, b), n in counts.items()}


def write_summary(reports: List[dict], matrix: Dict[tuple, float], output_dir: Path):
    """summary.md (scores + matrix) and summary_matrix.csv for the whole batch."""
    output_dir = Path(output_dir)
    reports = sorted(reports, key=lambda r: r["document_id"])
    assessed = [r["document_id"] for r in reports if r["total_integrity_risk"] != "NOT_ASSESSED"]

    lines = ["# Batch summary", "", "## Scores per document", "",
             "| Document | Risk | Peer % | External % | AI % |", "|---|---|---|---|---|"]
    for r in reports:
        s = r["scores"]
        lines.append(f"| {r['document_id']} | {r['total_integrity_risk']} | "
                     f"{_pct(s['peer_similarity_percentage'])} | "
                     f"{_pct(s['external_similarity_percentage'])} | "
                     f"{_pct(s['ai_generated_probability'])} |")

    lines += ["", "## Similarity matrix",
              "", "Each cell: percent of the ROW document's passages that match the COLUMN document.",
              "", "| | " + " | ".join(assessed) + " |", "|---|" + "---|" * len(assessed)]
    for a in assessed:
        cells = ["-" if a == b else f"{matrix.get((a, b), 0.0):.0f}" for b in assessed]
        lines.append(f"| **{a}** | " + " | ".join(cells) + " |")

    skipped = [r for r in reports if r["total_integrity_risk"] == "NOT_ASSESSED"]
    if skipped:
        lines += ["", "## Skipped files", ""]
        lines += [f"- {r['document_id']}: {r['warnings'][0]}" for r in skipped]
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    with open(output_dir / "summary_matrix.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["document"] + assessed)
        for a in assessed:
            writer.writerow([a] + ["" if a == b else round(matrix.get((a, b), 0.0), 1)
                                   for b in assessed])