import os

# Base paths
BASE_DIR = os.getcwd()

DATA_PATH = os.path.join(BASE_DIR, "data")
VECTOR_STORE_PATH = os.path.join(BASE_DIR, "vector-store")

# Chunking
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# OCR
OCR_MIN_TEXT_THRESHOLD = 50

# Files
INDEX_FILE = "index.faiss"
TEXTS_FILE = "texts.npy"
VECTORIZER_FILE = "vectorizer.pkl"
