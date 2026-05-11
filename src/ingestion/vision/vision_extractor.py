import base64
import os
import time

import requests
import requests.exceptions
from dotenv import load_dotenv

from ...core.config import (
    VISION_API_BASE_URL,
    VISION_INPUT_MIME_TYPE,
    VISION_MAX_RETRIES,
    VISION_MODEL_NAME,
    VISION_OCR_PROMPT,
    VISION_RATE_LIMIT_SECONDS,
    VISION_RETRY_SLEEP_SECONDS,
    VISION_TIMEOUT_SECONDS,
)

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

MIN_INTERVAL = VISION_RATE_LIMIT_SECONDS
LAST_CALL = 0


def rate_limit():
    global LAST_CALL

    now = time.time()

    if now - LAST_CALL < MIN_INTERVAL:
        time.sleep(MIN_INTERVAL - (now - LAST_CALL))

    LAST_CALL = time.time()


def to_base64(img_bytes):
    return base64.b64encode(img_bytes).decode("utf-8")


def extract_batch_images(images_bytes_list):
    if not images_bytes_list:
        return []

    rate_limit()

    parts = [
        {"text": VISION_OCR_PROMPT.strip()}
    ]

    for img in images_bytes_list:
        parts.append(
            {
                "inline_data": {
                    "mime_type": VISION_INPUT_MIME_TYPE,
                    "data": to_base64(img)
                }
            }
        )

    url = (
        f"{VISION_API_BASE_URL}/{VISION_MODEL_NAME}:generateContent"
        f"?key={GEMINI_API_KEY}"
    )

    payload = {"contents": [{"parts": parts}]}

    for attempt in range(VISION_MAX_RETRIES):
        try:
            res = requests.post(
                url,
                json=payload,
                timeout=VISION_TIMEOUT_SECONDS
            )

            if res.status_code == 429:
                time.sleep(VISION_RETRY_SLEEP_SECONDS)
                continue

            if res.status_code in (500, 502, 503, 504):
                print(f"[Retry {attempt+1}] Server busy ({res.status_code}), retrying...")
                time.sleep(VISION_RETRY_SLEEP_SECONDS)
                continue

            res.raise_for_status()
            break

        except (
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            requests.exceptions.ChunkedEncodingError,
            requests.exceptions.HTTPError,
        ):
            print(f"[Retry {attempt+1}] Connection failed, retrying...")
            time.sleep(VISION_RETRY_SLEEP_SECONDS)
    else:
        return [""] * len(images_bytes_list)

    data = res.json()

    if "candidates" not in data:
        return [""] * len(images_bytes_list)

    candidates = data.get("candidates", [])
    if not candidates:
        return [""] * len(images_bytes_list)

    content = candidates[0].get("content")
    if not content:
        return [""] * len(images_bytes_list)

    response_parts = content.get("parts", [])
    if not response_parts:
        return [""] * len(images_bytes_list)

    text = response_parts[0].get("text", "")

    split_outputs = []
    current = []

    for line in text.splitlines():
        if line.strip().lower().startswith("page"):
            if current:
                split_outputs.append("\n".join(current).strip())
                current = []
            continue
        current.append(line)

    if current:
        split_outputs.append("\n".join(current).strip())

    while len(split_outputs) < len(images_bytes_list):
        split_outputs.append("")

    return split_outputs[: len(images_bytes_list)]
