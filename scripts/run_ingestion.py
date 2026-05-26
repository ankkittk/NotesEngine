import os
import sys
import time

CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.core.config import ALLOWED_EXTENSIONS, DATA_PATH
from src.ingestion.ingest_service import ingest_single_file
from src.ingestion.ingestion_tracker import is_processed, load_tracker, save_tracker


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


def main():
    overall_start = time.perf_counter()

    processed_files = load_tracker()

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
        print("No new files to process.")
        return

    total = len(new_files)

    for idx, file_name in enumerate(new_files, start=1):
        file_path = os.path.join(DATA_PATH, file_name)
        started = time.perf_counter()

        render_progress(idx - 1, total, prefix="Ingesting", detail=file_name)

        result = ingest_single_file(
            file_path,
            processed_files=processed_files,
            save_tracker_after=False,
        )

        elapsed = time.perf_counter() - started

        if result["status"] == "ingested":
            print(
                f"[{idx}/{total}] {file_name} | "
                f"pages={result['pages']} "
                f"chunks={result['chunks']} "
                f"embeddings={result['embeddings']} | "
                f"{elapsed:.1f}s"
            )
        elif result["status"] == "skipped":
            print(
                f"[{idx}/{total}] {file_name} | skipped "
                f"({result['reason']}) | {elapsed:.1f}s"
            )
        elif result["status"] == "empty":
            print(
                f"[{idx}/{total}] {file_name} | empty "
                f"({result['reason']}) | {elapsed:.1f}s"
            )
        else:
            print(
                f"[{idx}/{total}] {file_name} | error "
                f"({result['reason']}) | {elapsed:.1f}s"
            )

        render_progress(idx, total, prefix="Ingesting", detail=file_name)

    save_tracker(processed_files)

    total_elapsed = time.perf_counter() - overall_start
    print(
        f"\nAdded {len(new_files)} new files | "
        f"total processed={len(processed_files)} | "
        f"done in {total_elapsed:.1f}s"
    )


if __name__ == "__main__":
    main()
