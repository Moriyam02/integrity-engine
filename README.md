# Multimodal Submission Integrity and Plagiarism Detection Engine

A command-line prototype that reads PDF, DOCX and short MP4 submissions, finds overlap between
them (including a video that reads out a document and text that has been reworded), compares them
with a local reference corpus, estimates how likely a text is AI-written, and writes an explainable
report for every file with the exact matching passages, page numbers and video timestamps.

## Quick start

Requires Python 3.9 or newer (developed and tested on Windows with Python 3.14; other versions are
untested).

```
python -m venv venv
venv\Scripts\activate            # Windows PowerShell: venv\Scripts\Activate.ps1   |   Mac/Linux: source venv/bin/activate
pip install -r requirements.txt && python download_models.py
python run_integrity_check.py --input_dir ./sample_submissions --output_dir ./reports
```

`download_models.py` downloads three open-source models once (about 700 MB in total). The last
command analyses the 15 sample files and writes, for each file, `<name>.json` and `<name>.md`, plus
`summary.md` and `summary_matrix.csv` for the whole batch.

Other commands:

```
python evaluate.py                  # compare reports with the ground truth (see EVALUATION.md)
python experiments.py               # compare embedding models and unit sizes
python tests/test_stage1.py         # unit tests (no model downloads needed)
```

## What it does

1. **Ingestion.** `pypdf` reads PDFs (page numbers kept; repeated headers, footers and page numbers
   removed on PDFs of three or more pages). `python-docx` reads Word files. `faster-whisper` (the
   open Whisper `tiny` model) transcribes the audio of MP4 files with segment timestamps. Empty,
   corrupted, scanned, silent or unsupported files are skipped with a warning and get a
   `NOT_ASSESSED` report; they never stop the batch.
2. **Chunking.** Text is cleaned and cut into overlapping chunks with their page or timestamp.
   Two sizes are used: 180-word chunks (about 235 tokens, inside the 200-400 token range of the
   brief) for the AI detector, and 40-word matching units with 10-word overlap for similarity.
3. **Embedding and index.** `paraphrase-MiniLM-L6-v2` (SentenceTransformers) turns every unit into a
   384-number vector, normalised so the inner product equals cosine similarity. Vectors are stored in
   a FAISS exact index (`IndexFlatIP`).
4. **Peer similarity.** Each unit is compared with all units of other files (a file is never compared
   with itself). A unit "matches" at cosine 0.75 or higher. A file's peer similarity is the share of
   its units that match; a single matching unit is treated as a coincidence, so at least two units
   must match.
5. **Template handling.** Text that appears almost identically (cosine 0.90 or higher) in at least
   three *other* files, such as a shared case prompt, is ignored in all scores.
6. **External similarity.** The same matching is run against `.txt` files in `reference_corpus/`, a
   local stand-in for web search (allowed by the brief).
7. **AI estimate.** A Hugging Face RoBERTa classifier (`Hello-SimpleAI/chatgpt-detector-roberta`)
   scores each 180-word chunk; the file score is the average. Texts under 40 words get no score.
8. **Reports.** Every match lists the passage from the checked file, the passage from the other
   file, file name, page or timestamp, cosine score and word-overlap (Jaccard) score. Cross-modal
   matches (video against document) are listed separately.

### Risk level

| Level | When |
|---|---|
| HIGH | peer similarity at or above 50 %, or external similarity at or above 50 %, or AI score at or above 90 % |
| MEDIUM | peer similarity at or above 20 %, or external at or above 20 %, or AI score at or above 70 % |
| LOW | otherwise |
| NOT_ASSESSED | the file could not be read |

These limits are heuristics set in `config.py` and tuned on the sample dataset (see
`EVALUATION.md` for the caveats).

### Report format

Each JSON report follows the structure in the brief: `document_id`, `total_integrity_risk`,
`scores` (`peer_similarity_percentage`, `external_similarity_percentage`,
`ai_generated_probability`, which is `null` when not available), `peer_matches`,
`multimodal_matches` and `explanations`. Each match has `matched_file`, `similarity_score`,
`query_passage`, `matched_passage` and a `location` (`query_page` or `query_timestamp`, and
`matched_page` or `matched_timestamp`). Three additions: `external_matches`, `warnings`, and
`word_overlap_jaccard` on every match (a low value next to a high similarity score shows the
text was reworded). The Markdown report shows the same information side by side. Real examples
are in `reports/`.

## Project structure

```
run_integrity_check.py     main command
evaluate.py                metrics against the ground truth
experiments.py             model and unit-size comparison
download_models.py         downloads the three models
config.py                  every setting and threshold
src/
  extractors.py            PDF / DOCX / MP4 to text, with error handling
  chunking.py              cleaning, chunks and matching units
  indexer.py               embeddings and FAISS index
  peer_similarity.py       file-to-file matching, self-match exclusion
  boilerplate.py           shared-template detection
  external_match.py        matching against the reference corpus
  ai_detector.py           AI-written text estimate
  reporter.py              JSON and Markdown reports, summary matrix
sample_submissions/        15 test files and dataset_manifest.json (ground truth)
reference_corpus/          local reference texts
reports/                   output of the sample run
tests/                     unit tests
```

## Assumptions

* Only `.pdf`, `.docx` and `.mp4` files are analysed; other types are skipped with a warning.
* PDFs must contain selectable text. Scanned PDFs are skipped with a warning (no OCR).
* Speech is English (`language="en"`). Videos with no audio track, or with no detected speech, are skipped.
* Word files have no fixed pages. Page numbers for DOCX are estimated from the page breaks saved
  by Word, or at about 450 words per page, and are marked as estimates in the Markdown report.
* Similarity is symmetric: both files in a copied pair are flagged. The tool cannot tell which is
  the original; a human reviewer decides.
* A passage is a template only if it appears in at least three other files, so two files that
  share a template are reported as matching.
* The external check uses the local `reference_corpus/` folder here, the public-domain poem and two articles(mount_everest, printing press) from wikipedia as the brief allows.
* Matching thresholds, the minimum of two matching units and the risk limits are heuristics tuned
  on the 15-file sample dataset; they are not validated on real submissions.
* The AI score is a statistical estimate and is advisory only.
* "The Last Leaf" by Oliver Wendell Holmes Sr. (1831) is public domain. All other sample texts and
  recordings were made by my ownself, except `ai_written.pdf` (written by Google
  Gemini) and the two articles in reference_corpus that have been taken from wikipedia.

## Scaling to 10,000 submissions

This section is a design, not something I have run: the prototype has only been run on 15 files,
and the figures below are back-of-the-envelope estimates that would need benchmarking.

**Assumed workload.** 10,000 submissions per challenge, about 90 % documents of about 1,500 words and
about 10 % videos of about 60 seconds. At 40-word units with 10-word overlap that is about
50 units per document and 5 per video transcript, so roughly **0.5 million vectors** of 384 dimensions
(about 0.8 GB as float32).

### Architecture

```
upload -> object storage (files) -> API / task database (status per submission)
   -> queue "extract"  (CPU workers: PDF, DOCX, chunking, content hash, template shingles)
   -> queue "asr"      (GPU workers: speech-to-text, videos only)
   -> queue "embed"    (GPU workers: batched embeddings)  -> vector database (HNSW)
   -> queue "compare"  (CPU: top-k search, scoring, Jaccard, risk)
   -> queue "ai"       (GPU or quantised CPU: AI detector)
   -> queue "report"   (CPU: JSON + Markdown) -> results database + report storage
```

* **Asynchronous queues.** One queue per stage (Celery with Redis or RabbitMQ, or the cloud
  equivalent such as SQS with container workers), so each stage scales on its own and a slow stage
  (speech-to-text) does not block the others. Tasks are idempotent and keyed by the SHA-256 of the
  file, so retries and duplicate uploads cost nothing. Failed tasks retry with backoff and then go to a
  dead-letter queue; a poison file is marked `NOT_ASSESSED`, as in the prototype, and the batch continues.
* **Compute.** Extraction, chunking, scoring and reporting are CPU work in autoscaled containers.
  GPUs are used for the three neural stages: speech-to-text (`faster-whisper`, `base` or `small` for
  better accuracy than `tiny`), embeddings (large batches) and the RoBERTa AI detector. A single
  mid-range GPU (T4/L4 class) per stage is enough for one challenge if the work is batched.
* **Order of work.** Documents are processed as they arrive, and each new submission is searched against
  what is already indexed (incremental comparison), so there is no big all-pairs job at the end. Only
  files in the same challenge are compared (a `challenge_id` filter), which also protects privacy.

### Vector database and index

* **Index type.** At 0.5 million vectors, exact brute-force search is still feasible: all-against-all
  is about 2 x 10^14 floating-point operations, roughly 20 seconds on one GPU or 10-20 minutes on a
  16-core CPU, if done in blocks. The prototype asks for all N neighbours of every unit, which only
  works for small N; production would ask for the top 10-50 per unit.
* **Recommendation: HNSW** (for example M=32, efConstruction=200, efSearch tuned to at least 95 %
  recall against an exact search on a sample). Query time grows roughly with log N, and memory is
  about 1 GB for 0.5 million vectors. It supports incremental inserts, which suits the arrival-order
  pipeline. Because the report shows scores as evidence, the candidates it returns are re-scored with
  exact cosine similarity before they are used.
* **Alternatives.** `IndexFlatIP` (exact, simplest) up to about a million vectors; IVF-PQ for
  hundreds of millions of vectors, where compressing 1,536 bytes per vector to tens of bytes matters
  more than the last few points of recall; sharding by challenge beyond that.
* **Database choice.** Qdrant (self-hosted, HNSW, filtering on `challenge_id`) or pgvector if the
  team already runs PostgreSQL and wants one system. Each vector stores its payload: file ID, page,
  timestamp and the unit text.
* **Templates at scale.** Instead of comparing every unit with every other, count repeated 8-word
  shingles across the challenge (document frequency); shingles found in at least k files are template text.
  This is linear in the amount of text.

### Throughput, cost and bottlenecks (estimates)

| Stage | Work (10,000 submissions) | Rough time on one worker | Note |
|---|---|---|---|
| Extraction and chunking | 9,000 documents | a few CPU-hours; minutes on 16 cores | CPU, parallel |
| Speech-to-text | about 1,000 minutes of audio | tens of minutes on one GPU; hours on CPU | biggest cost per item |
| Embeddings | about 0.5 million units | a few minutes on one GPU; 10-30 min on 16 cores | batch size 256 |
| AI detector | about 80,000 chunks of 512 tokens | 10-15 minutes on one GPU; hours on CPU | second biggest cost |
| Similarity search | top-k per unit against about 0.5 million | minutes (HNSW) | blocks of queries |

A full challenge should finish in roughly one to two hours of wall-clock time with one GPU and 16
CPU cores, at an on-demand cloud cost of the order of a few dollars to ten dollars per batch.
Neither figure is measured.

* **Latency.** A document submission can be reported within seconds to a minute once its stage is
  running (queue waiting dominates); a one-minute video takes a few minutes. Keep GPU workers warm to
  avoid model-loading delays.
* **Main bottlenecks.** Speech-to-text, the AI detector, very large or scanned PDFs (would need OCR
  and time limits), memory if a full similarity matrix were built, and hot queues at deadline time.
  Mitigations: autoscale on queue depth, batch GPU work, quantise or distil the detector, cap file size
  and processing time per file.
* **Operations and safety.** Run file parsers in a sandbox (untrusted uploads), store results and
  models with versions so scores can be reproduced, monitor queue depth, error rates and score
  distributions per stage, and apply retention and access rules because submissions are student data.
* **Changes to this codebase.** Replace the in-memory FAISS index with a persistent HNSW index or
  vector database; replace the all-neighbours search with top-k; replace the per-run template search with
  shingle counting; add the queues and a task-status table; keep the extractors, chunking, scoring and
  reporting modules as they are.

## Limitations

See `EVALUATION.md` for the measured results and failure analysis. In short: the dataset is small
and the settings were tuned on it, the AI detector missed text written by a newer chatbot (Gemini),
there is no OCR, and non-English or accented speech has not been tested.

## AI assistance and credits

 **Google Gemini** wrote `ai_written.pdf`, which is used as test input for the AI
detector.

Open-source components: faster-whisper (Whisper `tiny`), SentenceTransformers
(`paraphrase-MiniLM-L6-v2`; `all-MiniLM-L6-v2` and `all-mpnet-base-v2` only in `experiments.py`),
FAISS, `Hello-SimpleAI/chatgpt-detector-roberta`, pypdf, python-docx, NumPy. Wikipedia text, if
added to `reference_corpus/`, is CC BY-SA.