import os
import faiss
import pickle
import numpy as np

from config import VECTOR_STORE_PATH, INDEX_FILE, TEXTS_FILE, VECTORIZER_FILE

META_FILE = "metadata.pkl"


def load_vector_store():
    index = faiss.read_index(os.path.join(VECTOR_STORE_PATH, INDEX_FILE))

    texts = np.load(
        os.path.join(VECTOR_STORE_PATH, TEXTS_FILE),
        allow_pickle=True
    )

    with open(os.path.join(VECTOR_STORE_PATH, VECTORIZER_FILE), "rb") as f:
        vectorizer = pickle.load(f)

    with open(os.path.join(VECTOR_STORE_PATH, META_FILE), "rb") as f:
        metadata = pickle.load(f)

    return index, texts, vectorizer, metadata


def search(query, top_k=3):
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
