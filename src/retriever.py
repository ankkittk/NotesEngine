import os
import faiss
import pickle
import numpy as np

from config import VECTOR_STORE_PATH, INDEX_FILE, TEXTS_FILE, VECTORIZER_FILE


def load_vector_store():
    index = faiss.read_index(os.path.join(VECTOR_STORE_PATH, INDEX_FILE))

    texts = np.load(
        os.path.join(VECTOR_STORE_PATH, TEXTS_FILE),
        allow_pickle=True
    )

    with open(os.path.join(VECTOR_STORE_PATH, VECTORIZER_FILE), "rb") as f:
        vectorizer = pickle.load(f)

    return index, texts, vectorizer


def search(query, top_k=3):
    index, texts, vectorizer = load_vector_store()

    query_vec = vectorizer.transform([query]).toarray()

    D, I = index.search(np.array(query_vec), top_k)

    results = []
    for idx in I[0]:
        if idx < len(texts):
            results.append(texts[idx])

    return results
