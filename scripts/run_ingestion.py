import sys
import os

CURRENT_DIR = os.path.dirname(__file__)
SRC_PATH = os.path.abspath(os.path.join(CURRENT_DIR, "..", "src"))
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from config import DATA_PATH
from pdf_reader import extract_text_pages, needs_ocr
from ocr import ocr_document
from chunker import chunk_documents
from embedder import create_embeddings
from vector_store import store_embeddings


def process_pdf(file_path):
    pages = extract_text_pages(file_path)

    text_part = "\n".join(pages)

    # ALWAYS run OCR also
    ocr_part = ocr_document(file_path)

    # merge both
    return text_part + "\n" + ocr_part


def load_documents():
    documents = []

    for file in sorted(os.listdir(DATA_PATH)):
        if not file.lower().endswith(".pdf"):
            continue

        file_path = os.path.join(DATA_PATH, file)
        print(f"Processing {file}")

        text = process_pdf(file_path)

        if text.strip():
            documents.append(text)

    return documents


def main():
    documents = load_documents()

    if not documents:
        print("No PDFs found in data/")
        return

    chunks = chunk_documents(documents)

    if not chunks:
        print("No valid chunks produced")
        return

    embeddings, vectorizer = create_embeddings(chunks)
    store_embeddings(embeddings, chunks, vectorizer)


if __name__ == "__main__":
    main()
