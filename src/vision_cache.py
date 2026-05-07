import os
import json
import hashlib
from config import VECTOR_STORE_PATH

CACHE_PATH = os.path.join(VECTOR_STORE_PATH, "vision_cache.json")


def _load_cache():
    if not os.path.exists(CACHE_PATH):
        return {}
    try:
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return {}


def _save_cache(cache):
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f)


def _hash_bytes(data):
    return hashlib.md5(data).hexdigest()


def get_cached_result(img_bytes):
    cache = _load_cache()
    return cache.get(_hash_bytes(img_bytes))


def set_cached_result(img_bytes, text):
    cache = _load_cache()
    cache[_hash_bytes(img_bytes)] = text
    _save_cache(cache)
