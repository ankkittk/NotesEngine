import os
import base64
import requests
import time
from dotenv import load_dotenv
import requests.exceptions

from vision_cache import get_cached_result, set_cached_result

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

MIN_INTERVAL = 1.2
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
    outputs = []
    uncached_images = []
    uncached_indices = []

    # ---------------- CACHE CHECK ----------------
    for i, img in enumerate(images_bytes_list):
        cached = get_cached_result(img)

        if cached is not None:
            outputs.append(cached)

        else:
            outputs.append(None)
            uncached_images.append(img)
            uncached_indices.append(i)

    # nothing to call
    if not uncached_images:
        return outputs

    # ---------------- API CALL ----------------
    rate_limit()

    parts = [
        {
            "text": (
                "Extract ONLY visible text from each image.\n"
                "If no readable text is present, return EMPTY.\n"
                "Do NOT describe the image.\n"
                "Return strictly in this format:\n"
                "Page 1:\n...\n\nPage 2:\n...\n"
            )
        }
    ]

    for img in uncached_images:
        parts.append(
            {
                "inline_data": {
                    "mime_type": "image/png",
                    "data": to_base64(img)
                }
            }
        )

    url = (
        "https://generativelanguage.googleapis.com/v1/models/"
        "gemini-2.5-flash:generateContent"
        f"?key={GEMINI_API_KEY}"
    )

    payload = {"contents": [{"parts": parts}]}

    for attempt in range(3):
        try:
            res = requests.post(
                url,
                json=payload,
                timeout=20
            )

            if res.status_code == 429:
                time.sleep(2)
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
            time.sleep(2)

    else:
        return [""] * len(images_bytes_list)

    # ---------------- SAFE PARSING ----------------
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

    # ---------------- SPLIT OUTPUTS ----------------
    split_outputs = []
    current = []

    for line in text.splitlines():

        if line.strip().lower().startswith("page"):

            if current:
                split_outputs.append(
                    "\n".join(current).strip()
                )

                current = []

            continue

        current.append(line)

    if current:
        split_outputs.append(
            "\n".join(current).strip()
        )

    # ensure alignment
    while len(split_outputs) < len(uncached_images):
        split_outputs.append("")

    # ---------------- ASSIGN + CACHE ----------------
    for idx, out, img in zip(
        uncached_indices,
        split_outputs,
        uncached_images
    ):
        outputs[idx] = out
        set_cached_result(img, out)

    return outputs
