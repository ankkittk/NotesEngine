import os
import time

from .chunker import chunk_documents
from .document_loader import load_document
from .embedder import create_embeddings
from .ingestion_tracker import (
    is_processed,
    load_tracker,
    mark_processed,
    save_tracker,
)
from ..core.logger import log_ingestion
from ..retrieval.vector_store import store_embeddings


def ingest_single_file(
    file_path: str,
    processed_files=None,
    save_tracker_after: bool = True,
):
    if processed_files is None:
        processed_files = load_tracker()

    file_name = os.path.basename(file_path)

    started = time.perf_counter()

    if is_processed(file_name, processed_files):
        result = {
            "file_name": file_name,
            "status": "skipped",
            "reason": "already_processed",
            "pages": 0,
            "chunks": 0,
            "embeddings": 0,
            "elapsed_seconds": round(
                time.perf_counter() - started,
                2
            ),
        }

        log_ingestion(result)

        return result

    try:
        page_records = load_document(file_path)

        if not page_records:
            result = {
                "file_name": file_name,
                "status": "empty",
                "reason": "no_text_extracted",
                "pages": 0,
                "chunks": 0,
                "embeddings": 0,
                "elapsed_seconds": round(
                    time.perf_counter() - started,
                    2
                ),
            }

            log_ingestion(result)

            return result

        chunks = chunk_documents(page_records)

        if not chunks:
            result = {
                "file_name": file_name,
                "status": "empty",
                "reason": "no_valid_chunks",
                "pages": len(page_records),
                "chunks": 0,
                "embeddings": 0,
                "elapsed_seconds": round(
                    time.perf_counter() - started,
                    2
                ),
            }

            log_ingestion(result)

            return result

        embeddings, texts, metadata, vectorizer = (
            create_embeddings(chunks)
        )

        store_embeddings(
            embeddings=embeddings,
            texts=texts,
            metadata=metadata,
            vectorizer=vectorizer,
        )

        mark_processed(
            file_name,
            processed_files
        )

        if save_tracker_after:
            save_tracker(processed_files)

        result = {
            "file_name": file_name,
            "status": "ingested",
            "reason": "",
            "pages": len(page_records),
            "chunks": len(chunks),
            "embeddings": len(texts),
            "elapsed_seconds": round(
                time.perf_counter() - started,
                2
            ),
        }

        log_ingestion(result)

        return result

    except Exception as e:
        result = {
            "file_name": file_name,
            "status": "error",
            "reason": str(e),
            "pages": 0,
            "chunks": 0,
            "embeddings": 0,
            "elapsed_seconds": round(
                time.perf_counter() - started,
                2
            ),
        }

        log_ingestion(result)

        return result


def ingest_multiple_files(file_paths):
    processed_files = load_tracker()

    results = []

    for file_path in file_paths:
        result = ingest_single_file(
            file_path=file_path,
            processed_files=processed_files,
            save_tracker_after=False,
        )

        results.append(result)

    save_tracker(processed_files)

    return results
