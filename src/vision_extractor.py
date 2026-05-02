import os
import base64
import requests
import time
from dotenv import load_dotenv

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
    rate_limit()

    parts = [
        {
            "text": (
                "Extract text from EACH image separately.\n"
                "Return strictly in this format:\n"
                "Page 1:\n...\n\nPage 2:\n...\n"
            )
        }
    ]

    for img in images_bytes_list:
        parts.append({
            "inline_data": {
                "mime_type": "image/png",
                "data": to_base64(img)
            }
        })

    url = (
        "https://generativelanguage.googleapis.com/v1/models/"
        "gemini-2.5-flash:generateContent"
        f"?key={GEMINI_API_KEY}"
    )

    payload = {"contents": [{"parts": parts}]}

    # retry logic
    for _ in range(3):
        res = requests.post(url, json=payload)

        if res.status_code == 429:
            time.sleep(2)
            continue

        res.raise_for_status()
        break

    data = res.json()

    # -------- SAFE PARSING --------
    if "candidates" not in data:
        return [""] * len(images_bytes_list)

    candidates = data.get("candidates", [])
    if not candidates:
        return [""] * len(images_bytes_list)

    content = candidates[0].get("content")
    if not content:
        return [""] * len(images_bytes_list)

    parts = content.get("parts", [])
    if not parts:
        return [""] * len(images_bytes_list)

    text = parts[0].get("text", "")
    # --------------------------------

    # -------- SPLITTING LOGIC --------
    outputs = []
    current = []

    for line in text.splitlines():
        if line.strip().lower().startswith("page"):
            if current:
                outputs.append("\n".join(current).strip())
                current = []
            continue
        current.append(line)

    if current:
        outputs.append("\n".join(current).strip())

    # ensure alignment with input batch size
    while len(outputs) < len(images_bytes_list):
        outputs.append("")
    # --------------------------------

    return outputs
