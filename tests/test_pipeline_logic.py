"""Logic tests that need no model downloads.  Run with:  python tests/test_pipeline_logic.py
(also works with pytest)."""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import config
from src.boilerplate import find_boilerplate
from src.chunking import Chunk, chunk_by_sentences, location_label
from src.extractors import ExtractionResult, Segment, extract_file
from src.peer_similarity import find_peer_matches, peer_percentages
from src.reporter import _jaccard, assess_risk, build_report, build_skipped_report


class FakeIndex:
    """Exact cosine search with numpy, standing in for FAISS."""
    def __init__(self, vectors):
        self.v = vectors

    def search(self, queries, k):
        sims = queries @ self.v.T
        ids = np.argsort(-sims, axis=1)[:, :min(k, self.v.shape[0])]
        return np.take_along_axis(sims, ids, 1), ids


def _unit(rng, n=16):
    v = rng.normal(size=n)
    return (v / np.linalg.norm(v)).astype("float32")


def test_sentence_units_merge_short_and_keep_pages():
    text = "Hi. This sentence is long enough to stand alone in a unit. Another long enough sentence follows here."
    result = ExtractionResult("a.pdf", "pdf", segments=[Segment(text=text, page=3)])
    units = chunk_by_sentences(result, per_unit=1)
    assert len(units) == 2 and units[0].text.startswith("Hi. This")
    assert all(u.page_start == 3 for u in units)


def test_sentence_units_cut_very_long_text():
    result = ExtractionResult("a.docx", "docx", segments=[Segment(text="word " * 130)])
    assert len(chunk_by_sentences(result, per_unit=1, max_words=60)) == 3


def test_boilerplate_flagged_but_real_copy_is_not():
    rng = np.random.default_rng(0)
    template, copied = _unit(rng), _unit(rng)
    chunks, vecs = [], []
    for d in range(5):                                   # 5 docs share a template
        chunks.append(Chunk(f"d{d}", f"d{d}#0", "t", "docx")); vecs.append(template)
        chunks.append(Chunk(f"d{d}", f"d{d}#1", "x", "docx"))
        vecs.append(copied if d in (0, 4) else _unit(rng))   # d0 and d4 copy each other
    v = np.array(vecs)
    flagged = find_boilerplate(chunks, v, FakeIndex(v))
    assert sorted(chunks[i].chunk_id for i in flagged) == [f"d{d}#0" for d in range(5)]


def test_self_matches_are_excluded():
    rng = np.random.default_rng(1)
    same = _unit(rng)
    chunks = [Chunk("a", "a#0", "x", "docx"), Chunk("a", "a#1", "x", "docx")]  # identical, same doc
    v = np.array([same, same])
    matches = find_peer_matches(chunks, v, FakeIndex(v), set())
    assert matches["a"] == []


def test_lone_match_is_ignored_but_two_matches_count():
    rng = np.random.default_rng(2)
    shared = [_unit(rng) for _ in range(3)]
    chunks = [Chunk("a", f"a#{i}", "x", "docx") for i in range(3)] + \
             [Chunk("b", f"b#{i}", "y", "docx") for i in range(3)]
    # one shared unit only -> ignored
    vecs = [shared[0], _unit(rng), _unit(rng), shared[0], _unit(rng), _unit(rng)]
    v = np.array(vecs); m = find_peer_matches(chunks, v, FakeIndex(v), set())
    assert m["a"] == [] and peer_percentages(chunks, set(), m)["a"] == 0.0
    # two shared units -> counted
    vecs = [shared[0], shared[1], _unit(rng), shared[0], shared[1], _unit(rng)]
    v = np.array(vecs); m = find_peer_matches(chunks, v, FakeIndex(v), set())
    assert len(m["a"]) == 2 and abs(peer_percentages(chunks, set(), m)["a"] - 200 / 3) < 1e-6


def test_risk_levels():
    assert assess_risk(60, 0, 0) == "HIGH"
    assert assess_risk(25, 0, 0) == "MEDIUM"
    assert assess_risk(0, 0, 90) == "HIGH"
    assert assess_risk(0, 0, None) == "LOW"


def test_report_has_required_keys():
    report = build_report("a.pdf", 0.0, 0.0, None, [], [], 0)
    for key in ("document_id", "total_integrity_risk", "scores", "peer_matches",
                "multimodal_matches", "explanations"):
        assert key in report
    for key in ("peer_similarity_percentage", "external_similarity_percentage",
                "ai_generated_probability"):
        assert key in report["scores"]
    skipped = build_skipped_report("b.pdf", "file is empty (0 bytes)")
    assert skipped["total_integrity_risk"] == "NOT_ASSESSED" and skipped["warnings"]


def test_jaccard_low_for_reworded_text():
    assert _jaccard("the cat sat", "the cat sat") == 1.0
    assert _jaccard("remote work saves commute time", "staying home gives back hours") < 0.2


def test_docx_page_numbers_from_page_break():
    from docx import Document
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "two_pages.docx"
        doc = Document()
        doc.add_paragraph("First page paragraph with enough text to count.")
        doc.add_page_break()
        doc.add_paragraph("Second page paragraph with enough text to count.")
        doc.save(str(path))
        result = extract_file(path)
        pages = [s.page for s in result.segments]
        assert pages[0] == 1 and pages[-1] == 2


def test_location_label():
    assert location_label(Chunk("a", "a#0", "t", "pdf", page_start=2, page_end=2)) == "page 2"
    assert location_label(Chunk("a", "a#0", "t", "mp4", time_start=24, time_end=36)) == "00:00:24 - 00:00:36"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)