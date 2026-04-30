import sys
import os

CURRENT_DIR = os.path.dirname(__file__)
SRC_PATH = os.path.abspath(os.path.join(CURRENT_DIR, "..", "src"))
sys.path.append(SRC_PATH)

# Config
from config import DATA_PATH

# Modules
from ingestion.pdf_reader import extract_text_pages, needs_ocr
from ingestion.ocr import ocr_document
from ingestion.chunker import chunk_documents
from embedder import create_embeddings
from vector_store import store_embeddings


def process_pdf(file_path):
    pages = extract_text_pages(file_path)

    for text in pages:
        if needs_ocr(text):
            print(f"OCR triggered for {file_path}")
            return ocr_document(file_path)

    return "\n".join(pages)


def load_documents():
    documents = []

    for file in os.listdir(DATA_PATH):
        if file.endswith(".pdf"):
            file_path = os.path.join(DATA_PATH, file)
            print(f"Processing {file}")

            text = process_pdf(file_path)
            documents.append(text)

    return documents


def main():
    documents = load_documents()

    if not documents:
        print("No PDFs found in data/")
        return

    chunks = chunk_documents(documents)
    embeddings, vectorizer = create_embeddings(chunks)

    store_embeddings(embeddings, chunks, vectorizer)


if __name__ == "__main__":
    main()
