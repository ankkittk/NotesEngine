import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

DATA_PATH = os.path.join(BASE_DIR, "data")
VECTOR_STORE_PATH = os.path.join(BASE_DIR, "vector-db")

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

ALLOWED_EXTENSIONS = (".pdf", ".txt", ".docx", ".pptx")

INITIAL_RETRIEVAL_TOP_K = 20
RERANK_TOP_K = 3

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

GENERATION_MODEL_NAME = "openai/gpt-oss-120b"
GENERATION_BASE_URL = "https://api.groq.com/openai/v1"
GENERATION_TEMPERATURE = 0.2

OCR_MIN_TEXT_THRESHOLD = 5
VISION_BATCH_SIZE = 2
VISION_PDF_RENDER_SCALE = 1
VISION_RATE_LIMIT_SECONDS = 1.2
VISION_TIMEOUT_SECONDS = 20
VISION_MAX_RETRIES = 3
VISION_RETRY_SLEEP_SECONDS = 2
VISION_MODEL_NAME = "gemini-2.5-flash"
VISION_API_BASE_URL = "https://generativelanguage.googleapis.com/v1/models"
VISION_INPUT_MIME_TYPE = "image/png"

VISION_OCR_PROMPT = """
Extract ONLY visible text from each image.
If no readable text is present, return EMPTY.
Do NOT describe the image.
Return strictly in this format:
Page 1:
...

Page 2:
...
"""

INDEX_FILE = "index.faiss"
TEXTS_FILE = "texts.npy"
VECTORIZER_FILE = "vectorizer.pkl"
META_FILE = "metadata.json"

INDEX_PATH = os.path.join(VECTOR_STORE_PATH, INDEX_FILE)
TEXTS_PATH = os.path.join(VECTOR_STORE_PATH, TEXTS_FILE)
VECTORIZER_PATH = os.path.join(VECTOR_STORE_PATH, VECTORIZER_FILE)
META_PATH = os.path.join(VECTOR_STORE_PATH, META_FILE)

TRACKER_PATH = os.path.join(VECTOR_STORE_PATH, "processed_files.json")
