import os
import sys
import time

CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.core.config import DATA_PATH, ALLOWED_EXTENSIONS
from src.ingestion.chunker import chunk_documents
from src.ingestion.document_loader import load_document
from src.ingestion.embedder import create_embeddings
from src.ingestion.ingestion_tracker import (
    load_tracker,
    save_tracker,
    is_processed,
    mark_processed,
)
from src.retrieval.vector_store import store_embeddings


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

        page_records = load_document(file_path)
        elapsed = time.perf_counter() - started

        if page_records:
            documents.extend(page_records)
            file_names.append(file)

            combined_text = "\n\n".join(
                record.get("text", "") for record in page_records if record.get("text", "").strip()
            )
            word_count = len(combined_text.split())

            print(
                f"[{idx}/{total}] {file} | "
                f"pages={len(page_records)} words={word_count:,} chars={len(combined_text):,} | "
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

    print(f"\nChunking {len(documents)} page records...")
    chunk_start = time.perf_counter()
    chunks = chunk_documents(documents)
    print(f"Created {len(chunks)} chunks in {time.perf_counter() - chunk_start:.1f}s")

    if not chunks:
        print("No valid chunks produced")
        return

    print("Creating embeddings...")
    emb_start = time.perf_counter()
    embeddings, texts, metadata, vectorizer = create_embeddings(chunks)
    print(f"Embeddings ready in {time.perf_counter() - emb_start:.1f}s")

    print("Updating vector store...")
    store_embeddings(embeddings, texts, metadata, vectorizer)

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
