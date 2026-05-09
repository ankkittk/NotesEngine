import os
import pickle

import faiss
import numpy as np

from ..core.config import INITIAL_RETRIEVAL_TOP_K, INDEX_PATH, META_PATH, TEXTS_PATH, VECTORIZER_PATH


def _fallback_metadata(texts):
    return [
        {
            "text": text,
            "source": "unknown",
            "chunk_id": f"legacy_{i}"
        }
        for i, text in enumerate(texts)
    ]


def load_vector_store():
    index = faiss.read_index(INDEX_PATH)

    texts = np.load(
        TEXTS_PATH,
        allow_pickle=True
    )

    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)

    if os.path.exists(META_PATH):
        with open(META_PATH, "rb") as f:
            metadata = pickle.load(f)
    else:
        metadata = _fallback_metadata(texts.tolist())

    return index, texts, vectorizer, metadata


def search(query, top_k=INITIAL_RETRIEVAL_TOP_K):
    index, texts, vectorizer, metadata = load_vector_store()

    query_vec = vectorizer.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    _, I = index.search(np.array(query_vec), top_k)

    results = []
    for idx in I[0]:
        if 0 <= idx < len(metadata):
            results.append(metadata[idx]["text"])

    return results
