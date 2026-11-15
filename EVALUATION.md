# Evaluation

All numbers below come from running `python run_integrity_check.py ...`, `python evaluate.py` and
`python experiments.py` on the 15 files in `sample_submissions/` (raw output is saved in
`reports/evaluation_results.txt` and `reports/experiment_results.txt`).

 The dataset is small (13 readable files, 4 truly matching pairs, 1 AI-written file). 
 The embedding model, similarity threshold and AI-score limits were chosen after looking at results on this same dataset, so the final metrics are optimistic. There is no held-out test set. The results show the pipeline behaves correctly on
these scenarios; they do not estimate accuracy on real submissions.

## 1. Dataset and ground truth

15 files, listed with their ground truth in `sample_submissions/dataset_manifest.json`.

| Group | Files | Ground truth |
|---|---|---|
| Poem copy group | `last_leaf.docx`, `last_leaf_copy.pdf`, `last_leaf_reading.mp4` | Same text in three forms (Word, PDF, a spoken reading of the first two stanzas). All 3 pairs are real matches. The poem also exists in `reference_corpus/` (external match). |
| Reworded pair | `para_a.docx`, `para_b.docx` | Same ideas, reworded by hand. 1 real match. |
| Shared-prompt group | `case_01.docx`, `case_02.docx`, `case_03.pdf`, `case_04.pdf`, `case_05.pdf` | Identical case prompt, different original answers. No real matches (Scenario 4). |
| AI-written | `ai_written.pdf` | Written by Google Gemini. |
| Independent originals | `original_essay.pdf`, `original_pitch.mp4` | Own writing and speech, unique topics. |
| Degraded | `empty.pdf` (0 bytes), `silent.mp4` (phone video with no audio) | Must be skipped with a warning. |

Formats among readable files: 6 PDF, 5 DOCX, 2 spoken MP4. With 13 readable files there are
78 file pairs, of which 4 are real matches.

**Authorship.** "The Last Leaf" (Oliver Wendell Holmes Sr., 1831) is public domain. All other
texts and recordings were written or recorded by my ownself, except
`ai_written.pdf`as it was taken from google gemini and the two articles in reference_corpus (mount_everest, printing_press) that were taken from Wikipedia. The AI signal is therefore evaluated only on files with a known label: 1 AI-written
file (positive) and 12 human files (negatives).

## 2. How results are measured

* **Pair level (Part A of `evaluate.py`).** For every pair of readable files, the score is the best
  cosine similarity between any matching unit of one file and any of the other (template text
  excluded). A pair is predicted "match" if the score is at or above a threshold.
* **File level (Part B).** A file is "flagged" if its final risk is MEDIUM or HIGH. Ground truth:
  the copy group, the reworded pair and the AI-written file should be flagged; the shared-prompt
  group and the independent originals should not. Both files of a copy pair are flagged, because the
  system cannot tell which one is the original.
* **Per-signal confusion matrices** for peer similarity, external similarity and AI probability,
  using the thresholds that drive the risk level (peer 20 %, external 20 %, AI 70 %).

## 3. Results with the final settings

Final settings (`config.py`): embedding model `paraphrase-MiniLM-L6-v2`, 40-word matching units with
10-word overlap, similarity threshold 0.75, at least 2 matching units per file, template rule
"near-identical text in at least 3 other files".

### 3.1 Pair level

Best scores, highest first:

| Score | Truth | Pair |
|---|---|---|
| 1.00 | MATCH | `last_leaf.docx` and `last_leaf_copy.pdf` |
| 0.96 | MATCH | `last_leaf.docx` and `last_leaf_reading.mp4` |
| 0.96 | MATCH | `last_leaf_copy.pdf` and `last_leaf_reading.mp4` |
| 0.95 | MATCH | `para_a.docx` and `para_b.docx` |
| 0.57 | no | `case_01.docx` and `case_03.pdf` (same-case answers) |
| 0.55 | no | `case_02.docx` and `case_03.pdf` |
| 0.54 | no | `case_03.pdf` and `case_05.pdf` |

Lowest score among real matches: **0.95**. Highest among non-matches: **0.57**. Every threshold from
0.60 to 0.95 separates them perfectly on this dataset (TP 4, FP 0, FN 0, TN 74); the chosen 0.75 sits
in the middle of that gap. The shared case prompt produced 24 template units, which were ignored.

### 3.2 File level

| File | Expected | Got | Peer % | External % | AI % |
|---|---|---|---|---|---|
| `last_leaf.docx` | flagged | HIGH | 100.0 | 100.0 | 2.9 |
| `last_leaf_copy.pdf` | flagged | HIGH | 100.0 | 100.0 | 2.9 |
| `last_leaf_reading.mp4` | flagged | HIGH | 100.0 | 100.0 | 2.2 |
| `para_a.docx` | flagged | HIGH | 100.0 | 0.0 | 0.0 |
| `para_b.docx` | flagged | HIGH | 100.0 | 0.0 | 0.0 |
| `case_01.docx` | clean | LOW | 0.0 | 0.0 | 0.1 |
| `case_02.docx` | clean | LOW | 0.0 | 0.0 | 0.4 |
| `case_03.pdf` | clean | LOW | 0.0 | 0.0 | 0.1 |
| `case_04.pdf` | clean | LOW | 0.0 | 0.0 | 0.0 |
| `case_05.pdf` | clean | LOW | 0.0 | 0.0 | 0.1 |
| `ai_written.pdf` | flagged | **LOW (missed)** | 0.0 | 0.0 | 0.0 |
| `original_essay.pdf` | clean | LOW | 0.0 | 0.0 | 51.8 |
| `original_pitch.mp4` | clean | LOW | 0.0 | 0.0 | 0.6 |
| `empty.pdf` | skipped | NOT_ASSESSED | - | - | - |
| `silent.mp4` | skipped | NOT_ASSESSED | - | - | - |

Both unreadable files were skipped with a clear warning and the rest of the batch was processed
(2 of 2).

### 3.3 Confusion matrices (13 readable files)

Overall (flagged = MEDIUM or HIGH): **precision 1.00, recall 0.83, F1 0.91**

| | Predicted flagged | Predicted clean |
|---|---|---|
| Actually flagged | 5 | 1 |
| Actually clean | 0 | 7 |

Peer signal (peer % >= 20): precision 1.00, recall 1.00 (TP 5, FP 0, FN 0, TN 8).
External signal (external % >= 20): precision 1.00, recall 1.00 (TP 3, FP 0, FN 0, TN 10).
AI signal (AI % >= 70): TP 0, FP 0, FN 1, TN 12. Recall 0.00; precision is undefined because the
detector never predicted "AI".

### 3.4 Before and after tuning

| | First run (`all-MiniLM-L6-v2`, threshold 0.65, AI limit 50) | Final |
|---|---|---|
| Pair level: false pairs at the threshold | 6 | 0 |
| Best score of an innocent pair | 0.94 | 0.57 |
| File level precision / recall / F1 | 0.83 / 0.83 / 0.83 | 1.00 / 0.83 / 0.91 |
| AI detector | 1 miss, 1 false alarm (`original_essay.pdf`, 51.8 %) | 1 miss, 0 false alarms |

## 4. Experiment: unit size and embedding model

`python experiments.py` classifies file pairs for every combination of three embedding models and
five ways of cutting text into matching units. "Real-min" is the lowest best-score among real
matches, "benign-max" the highest among all other pairs. A larger gap (real-min minus benign-max)
means a safer threshold exists.

| Model | Units | Real-min | Benign-max | Gap | Best threshold | TP / FP / FN | F1 |
|---|---|---|---|---|---|---|---|
| all-MiniLM-L6-v2 | words 40/10 | 0.96 | 0.94 | +0.02 | 0.75 | 4 / 0 / 0 | 1.00 |
| all-MiniLM-L6-v2 | words 80/20 | 0.85 | 0.77 | +0.08 | 0.80 | 4 / 0 / 0 | 1.00 |
| all-MiniLM-L6-v2 | sentences x1 | 0.72 | 0.79 | -0.07 | 0.70 | 4 / 0 / 0 | 1.00 |
| all-MiniLM-L6-v2 | sentences x2 | 0.78 | 0.79 | -0.01 | 0.70 | 4 / 1 / 0 | 0.89 |
| all-MiniLM-L6-v2 | sentences x3 | 0.66 | 0.80 | -0.14 | 0.65 | 2 / 0 / 2 | 0.67 |
| paraphrase-MiniLM-L6-v2 | **words 40/10** | **0.95** | **0.57** | **+0.38** | 0.75 | 4 / 0 / 0 | 1.00 |
| paraphrase-MiniLM-L6-v2 | words 80/20 | 0.89 | 0.80 | +0.09 | 0.80 | 4 / 0 / 0 | 1.00 |
| paraphrase-MiniLM-L6-v2 | sentences x1 | 0.74 | 0.67 | +0.07 | 0.70 | 4 / 0 / 0 | 1.00 |
| paraphrase-MiniLM-L6-v2 | sentences x2 | 0.77 | 0.66 | +0.11 | 0.70 | 4 / 0 / 0 | 1.00 |
| paraphrase-MiniLM-L6-v2 | sentences x3 | 0.80 | 0.83 | -0.03 | 0.70 | 2 / 0 / 2 | 0.67 |
| all-mpnet-base-v2 | words 40/10 | 0.96 | 0.87 | +0.09 | 0.80 | 4 / 0 / 0 | 1.00 |
| all-mpnet-base-v2 | words 80/20 | 0.87 | 0.84 | +0.03 | 0.85 | 4 / 0 / 0 | 1.00 |
| all-mpnet-base-v2 | sentences x1 | 0.79 | 0.71 | +0.08 | 0.65 | 4 / 0 / 0 | 1.00 |
| all-mpnet-base-v2 | sentences x2 | 0.79 | 0.73 | +0.06 | 0.65 | 4 / 1 / 0 | 0.89 |
| all-mpnet-base-v2 | sentences x3 | 0.72 | 0.70 | +0.02 | 0.65 | 2 / 0 / 2 | 0.67 |

What this shows:

* With only 4 real pairs, most settings reach F1 1.00, so F1 alone cannot choose between them. The
  gap is more informative, and `paraphrase-MiniLM-L6-v2` with 40-word units has by far the largest.
* My expectation that whole-sentence units would line up better with reworded text was **not**
  supported: they were no better than word windows, and units of three sentences lost real matches
  (FN 2) because too much unrelated text diluted the score. Sentence units remain available in the
  code (`MATCH_UNIT = "sentences"`) but are not used.
* Early in development (on a different practice recording), comparing 180-word chunks gave a cosine
  of 0.52 for a spoken excerpt against the poem document, while 40-word windows gave 0.80. This is
  why matching uses small units while the 180-word chunks (inside the PDF's 200-400 token range)
  are used only for the AI detector.

## 5. Failure analysis

**Failure 1: the AI detector missed text written by Gemini (`ai_written.pdf`, 0.0 %).**
The detector (`Hello-SimpleAI/chatgpt-detector-roberta`) was trained on early-2023 ChatGPT answers,
so text from newer chatbots is probably out of distribution. This is my hypothesis; I did not test
other detectors. *How I would fix it:* evaluate newer or multiple detectors on a labelled set that
includes current chatbots, combine the classifier with perplexity/burstiness features, and
calibrate the output. Until then the AI score should be treated as advisory.

**Failure 2: the AI detector is unreliable near 50 % (`original_essay.pdf`, 51.8 % on honest human
writing).** With the original limit (MEDIUM at 50 %) this caused a false alarm. I raised the limits
to 70 % (MEDIUM) and 90 % (HIGH), which removed the false alarm but did not help recall. This change
was made after seeing the result, so it is tuning on the test set. *Better fix:* a larger labelled set
to calibrate the limits, and never letting the AI score alone trigger a penalty.

**Failure 3: honest answers to the same prompt can look highly similar to a general-purpose
embedding model.** With `all-MiniLM-L6-v2`, six pairs of independent answers to the same case
scored between 0.79 and 0.94 on their best single unit. No file was flagged only because of the rule
that a file needs at least 2 matching units. The likely cause is shared topic vocabulary, possibly
combined with units that straddle the end of the shared prompt and the start of an answer (which
the template filter does not catch). I did not verify this cause; switching to
`paraphrase-MiniLM-L6-v2` lowered the best innocent score to 0.57. *Better fix:* strip template text
(found by counting repeated word n-grams across files) before cutting the text into units, and
test on a larger set of same-topic honest submissions.

**Failure 4: speech-to-text errors and coarse timestamps (observed during development).**
`whisper-tiny` made small errors on the poem (for example "air the pruning knife" for "Ere the
pruning-knife"), and the similarity still passed the threshold, but each video match only carries
the time range of the transcript segment it came from (segments of 10-20 seconds), so two
windows can show overlapping timestamps. *Better fix:* use word-level timestamps from the
speech model and a larger model (`base` or `small`), and evaluate on accented speech, which this
dataset does not include.

## 6. Trade-off analysis: embedding model and vector index

* **Embedding model.** All three candidates produce 384 (MiniLM) or 768 (MPNet) dimensional vectors.
  The two MiniLM models download as about 91 MB each, and `all-mpnet-base-v2` as about 438 MB
  (about 4.8 times larger, so slower to run and needing roughly twice the vector storage). I did not
  benchmark speed. `paraphrase-MiniLM-L6-v2` was chosen because it gave the best separation
  at the smallest size, which suits CPU-only machines and the PDF's request for a lightweight model.
  `all-MiniLM-L6-v2` produced more false similarity between same-topic answers, and the larger MPNet
  model did not beat it enough to justify the cost on this dataset.
* **Vector index.** The prototype uses FAISS `IndexFlatIP` on normalised vectors, which is exact
  cosine similarity. With about 300 units in the sample set, an approximate index would only add
  complexity and a recall risk. For 10,000 submissions see the scaling plan in `README.md`
  (HNSW plus exact re-ranking of the best candidates).
* **Chunk size.** 40-word matching units with 10-word overlap found partial copies (a video reading
  two stanzas) and reworded text; 180-word chunks diluted partial copies. Larger units cost fewer
  vectors but lose precision; smaller units gain precision but cost more vectors and more
  accidental matches.

## 7. Other limitations

* Only 4 real matching pairs and 1 AI-written file: precision and recall have very wide error bars.
* Mostly short, author-written texts; the cross-modal test is a poem, which is verse and not typical
  prose. No scanned PDFs, no accented speech, no languages other than English.
* The external reference corpus is tiny (the poem), so external false positives are barely tested.
* A passage shared by exactly two files is never treated as a template, and a passage shared by
  three or more files that were truly copied from each other would be wrongly ignored.
* DOCX page numbers are estimates, not true pages.
* Thresholds were tuned on the same data that produced the metrics (see the top of this file).

## 8. Reproduce

```
python run_integrity_check.py --input_dir ./sample_submissions --output_dir ./reports
python evaluate.py
python experiments.py
```