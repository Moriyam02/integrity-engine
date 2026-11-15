"""All settings in one place so they are easy to explain and change."""

# --- Official chunks (Task 1): ~235 tokens, inside the PDF's 200-400 range and
# --- under the 256-token limit of the embedding model. Used by the AI detector.
CHUNK_WORDS = 180
CHUNK_OVERLAP_WORDS = 30

# --- Matching units: the small pieces that are compared across documents ---
MATCH_UNIT = "words"          # "words" (sliding windows) or "sentences" (whole sentences)
MATCH_WINDOW_WORDS = 40       # used when MATCH_UNIT = "words"
MATCH_WINDOW_OVERLAP = 10
MATCH_SENTENCES = 1           # sentences per unit when MATCH_UNIT = "sentences"
MIN_MATCHED_UNITS = 2         # a single matching unit is treated as a coincidence

# --- Speech to text ---
WHISPER_MODEL = "tiny"        # use "base" for better accuracy, but slower
MIN_TRANSCRIPT_WORDS = 5      # fewer words than this = treat the video as silent

# --- PDF header/footer removal ---
REPEATED_LINE_RATIO = 0.6     # a line on this share of pages is a header/footer
MIN_PAGES_FOR_HEADER_DETECTION = 3

# --- DOCX page estimate (Word files store no fixed pages) ---
DOCX_WORDS_PER_PAGE = 450

# --- Embeddings ---
EMBEDDING_MODEL = "sentence-transformers/paraphrase-MiniLM-L6-v2"

# --- Similarity thresholds (chosen from the experiments in EVALUATION.md) ---
SIMILARITY_THRESHOLD = 0.75           # peer matching: cosine score for "same passage"
EXTERNAL_SIMILARITY_THRESHOLD = 0.75  # matching against the reference corpus
BOILERPLATE_SIMILARITY = 0.90         # near-identical text...
BOILERPLATE_MIN_DOCS = 3              # ...in this many OTHER documents = template

# --- External / web reference similarity (local knowledge base, as the PDF allows) ---
REFERENCE_DIR = "./reference_corpus"

# --- AI-generated text detection ---
AI_DETECTOR_MODEL = "Hello-SimpleAI/chatgpt-detector-roberta"
AI_LABEL = "ChatGPT"          # the label that means "AI-written"
AI_MIN_WORDS = 40             # shorter pieces are too short to judge reliably

# --- Overall risk: HIGH if ANY score reaches the HIGH limit, MEDIUM if any reaches MEDIUM ---
RISK_HIGH = {"peer": 50.0, "external": 50.0, "ai": 90.0}
RISK_MEDIUM = {"peer": 20.0, "external": 20.0, "ai": 70.0}

# --- Markdown report ---
MD_MAX_MATCHES = 10           # matches shown per section (all are in the JSON)

# --- Folders ---
DEFAULT_INPUT_DIR = "./sample_submissions"
DEFAULT_OUTPUT_DIR = "./reports"