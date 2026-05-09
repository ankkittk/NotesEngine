import os
import pickle

import faiss
import numpy as np

from ..core.config import INDEX_PATH, META_PATH, TEXTS_PATH, VECTOR_STORE_PATH, VECTORIZER_PATH


def _fallback_metadata(texts):
    return [
        {
            "text": text,
            "source": "unknown",
            "chunk_id": f"legacy_{i}"
        }
        for i, text in enumerate(texts)
    ]


def store_embeddings(embeddings, texts, metadata, vectorizer):
    os.makedirs(VECTOR_STORE_PATH, exist_ok=True)

    embeddings = np.asarray(embeddings, dtype="float32")

    if os.path.exists(INDEX_PATH):
        index = faiss.read_index(INDEX_PATH)

        existing_texts = []
        if os.path.exists(TEXTS_PATH):
            existing_texts = np.load(TEXTS_PATH, allow_pickle=True).tolist()

        if os.path.exists(META_PATH):
            with open(META_PATH, "rb") as f:
                existing_meta = pickle.load(f)
        else:
            existing_meta = _fallback_metadata(existing_texts)

        existing_texts.extend(texts)
        existing_meta.extend(metadata)

        index.add(embeddings)

        faiss.write_index(index, INDEX_PATH)
        np.save(TEXTS_PATH, np.array(existing_texts, dtype=object), allow_pickle=True)

        with open(META_PATH, "wb") as f:
            pickle.dump(existing_meta, f)

        print("\nAppended to existing vector store")

    else:
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)

        faiss.write_index(index, INDEX_PATH)
        np.save(TEXTS_PATH, np.array(texts, dtype=object), allow_pickle=True)

        with open(META_PATH, "wb") as f:
            pickle.dump(metadata, f)

        print("\nCreated new vector store")

    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)
