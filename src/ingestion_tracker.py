import os
import json
from config import TRACKER_PATH


def load_tracker():
    if not os.path.exists(TRACKER_PATH):
        return set()

    try:
        with open(TRACKER_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            return set(data) if isinstance(data, list) else set()
    except (json.JSONDecodeError, IOError):
        # corrupted or unreadable file → reset safely
        return set()


def save_tracker(processed_files):
    os.makedirs(os.path.dirname(TRACKER_PATH), exist_ok=True)

    with open(TRACKER_PATH, "w", encoding="utf-8") as f:
        json.dump(sorted(list(processed_files)), f, indent=2)


def is_processed(file_name, processed_files):
    return file_name in processed_files


def mark_processed(file_name, processed_files):
    processed_files.add(file_name)
