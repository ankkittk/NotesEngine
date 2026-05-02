import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DATA_PATH = os.path.join(BASE_DIR, "data")
VECTOR_STORE_PATH = os.path.join(BASE_DIR, "vector-db")

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
OCR_MIN_TEXT_THRESHOLD = 50

ALLOWED_EXTENSIONS = (".pdf", ".txt", ".docx", ".pptx")

INDEX_FILE = "index.faiss"
TEXTS_FILE = "texts.npy"
VECTORIZER_FILE = "vectorizer.pkl"
