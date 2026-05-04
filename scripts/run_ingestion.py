import sys
import os
import time

CURRENT_DIR = os.path.dirname(__file__)
SRC_PATH = os.path.abspath(os.path.join(CURRENT_DIR, "..", "src"))
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from config import DATA_PATH, ALLOWED_EXTENSIONS
from document_loader import load_document
from chunker import chunk_documents
from embedder import create_embeddings
from vector_store import store_embeddings
from ingestion_tracker import (
    load_tracker,
    save_tracker,
    is_processed,
    mark_processed,
)


def render_progress(current, total, prefix="", detail=""):
    total = max(total, 1)
    bar_len = 24
    filled = int(bar_len * current / total)
    bar = "█" * filled + "-" * (bar_len - filled)
    pct = (current / total) * 100

    sys.stdout.write(
        f"\r{prefix} |{bar}| {current}/{total} ({pct:5.1f}%) {detail}"
    )
    sys.stdout.flush()

    if current >= total:
        sys.stdout.write("\n")


def load_new_documents(processed_files):
    documents = []
    file_names = []

    all_files = [
        f for f in sorted(os.listdir(DATA_PATH))
        if f.lower().endswith(ALLOWED_EXTENSIONS)
    ]
    new_files = [f for f in all_files if not is_processed(f, processed_files)]

    print(
        f"Found {len(all_files)} supported files | "
        f"{len(processed_files)} already processed | "
        f"{len(new_files)} new to ingest"
    )

    if not new_files:
        return documents, file_names

    total = len(new_files)

    for idx, file in enumerate(new_files, start=1):
        file_path = os.path.join(DATA_PATH, file)
        started = time.perf_counter()

        render_progress(idx - 1, total, prefix="Ingesting", detail=file)

        text = load_document(file_path)
        elapsed = time.perf_counter() - started

        if text.strip():
            documents.append(text)
            file_names.append(file)
            word_count = len(text.split())
            print(
                f"[{idx}/{total}] {file} | "
                f"words={word_count:,} chars={len(text):,} | "
                f"{elapsed:.1f}s"
            )
        else:
            print(f"[{idx}/{total}] {file} | empty | {elapsed:.1f}s")

        render_progress(idx, total, prefix="Ingesting", detail=file)

    return documents, file_names


def main():
    overall_start = time.perf_counter()

    processed_files = load_tracker()
    documents, file_names = load_new_documents(processed_files)

    if not documents:
        print("No new files to process.")
        return

    print(f"\nChunking {len(documents)} documents...")
    chunk_start = time.perf_counter()
    chunks = chunk_documents(documents)
    print(f"Created {len(chunks)} chunks in {time.perf_counter() - chunk_start:.1f}s")

    if not chunks:
        print("No valid chunks produced")
        return

    print("Creating embeddings...")
    emb_start = time.perf_counter()
    embeddings, vectorizer = create_embeddings(chunks)
    print(f"Embeddings ready in {time.perf_counter() - emb_start:.1f}s")

    print("Updating vector store...")
    store_embeddings(embeddings, chunks, vectorizer)

    for f in file_names:
        mark_processed(f, processed_files)

    save_tracker(processed_files)

    total_elapsed = time.perf_counter() - overall_start
    print(
        f"\nAdded {len(file_names)} new files | "
        f"total processed={len(processed_files)} | "
        f"done in {total_elapsed:.1f}s"
    )


if __name__ == "__main__":
    main()
