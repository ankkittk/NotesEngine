import json
import os
import pickle

import faiss
import numpy as np

from ..core.config import INITIAL_RETRIEVAL_TOP_K, INDEX_PATH, META_PATH, TEXTS_PATH, VECTORIZER_PATH


def _normalize_metadata_item(item):
    if not isinstance(item, dict):
        return {
            "source": "unknown",
            "page": None,
            "chunk_id": None,
        }

    return {
        "source": item.get("source", "unknown"),
        "page": item.get("page"),
        "chunk_id": item.get("chunk_id"),
    }


def _load_metadata():
    if os.path.exists(META_PATH):
        with open(META_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [_normalize_metadata_item(item) for item in data] if isinstance(data, list) else []

    legacy_meta_path = os.path.join(os.path.dirname(META_PATH), "metadata.pkl")
    if os.path.exists(legacy_meta_path):
        with open(legacy_meta_path, "rb") as f:
            data = pickle.load(f)
        return [_normalize_metadata_item(item) for item in data] if isinstance(data, list) else []

    return []


def load_vector_store():
    index = faiss.read_index(INDEX_PATH)

    texts = np.load(
        TEXTS_PATH,
        allow_pickle=True
    ).tolist()

    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)

    metadata = _load_metadata()

    if len(texts) != len(metadata):
        raise ValueError("Vector store is inconsistent: texts and metadata lengths differ.")

    return index, texts, vectorizer, metadata


def search(query, top_k=INITIAL_RETRIEVAL_TOP_K):
    index, texts, vectorizer, metadata = load_vector_store()

    query_vec = vectorizer.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    distances, indices = index.search(np.array(query_vec), top_k)

    results = []
    for rank, idx in enumerate(indices[0]):
        if idx < 0 or idx >= len(texts):
            continue

        meta = metadata[idx] if idx < len(metadata) else {}
        results.append(
            {
                "text": texts[idx],
                "source": meta.get("source", "unknown"),
                "page": meta.get("page"),
                "chunk_id": meta.get("chunk_id"),
                "retrieval_distance": float(distances[0][rank]),
            }
        )

    return results
