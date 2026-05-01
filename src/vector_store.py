import os
import numpy as np
import faiss
import pickle

from config import VECTOR_STORE_PATH, INDEX_FILE, TEXTS_FILE, VECTORIZER_FILE


def store_embeddings(embeddings, chunks, vectorizer):
    os.makedirs(VECTOR_STORE_PATH, exist_ok=True)

    embeddings = np.asarray(embeddings, dtype="float32")
    dim = embeddings.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    faiss.write_index(index, os.path.join(VECTOR_STORE_PATH, INDEX_FILE))
    np.save(os.path.join(VECTOR_STORE_PATH, TEXTS_FILE), np.array(chunks, dtype=object), allow_pickle=True)

    with open(os.path.join(VECTOR_STORE_PATH, VECTORIZER_FILE), "wb") as f:
        pickle.dump(vectorizer, f)

    print("\nVector store created successfully")
