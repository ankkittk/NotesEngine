import os
import numpy as np
import faiss
import pickle

from config import VECTOR_STORE_PATH, INDEX_FILE, TEXTS_FILE, VECTORIZER_FILE


def store_embeddings(embeddings, chunks, vectorizer):
    os.makedirs(VECTOR_STORE_PATH, exist_ok=True)

    index_path = os.path.join(VECTOR_STORE_PATH, INDEX_FILE)
    texts_path = os.path.join(VECTOR_STORE_PATH, TEXTS_FILE)
    vectorizer_path = os.path.join(VECTOR_STORE_PATH, VECTORIZER_FILE)

    embeddings = np.asarray(embeddings, dtype="float32")

    # APPEND MODE
    if os.path.exists(index_path):
        index = faiss.read_index(index_path)

        existing_texts = np.load(texts_path, allow_pickle=True).tolist()
        existing_texts.extend(chunks)

        index.add(embeddings)

        faiss.write_index(index, index_path)
        np.save(texts_path, np.array(existing_texts, dtype=object), allow_pickle=True)

        print("\nAppended to existing vector store")

    # FIRST-TIME MODE
    else:
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)

        faiss.write_index(index, index_path)
        np.save(texts_path, np.array(chunks, dtype=object), allow_pickle=True)

        print("\nCreated new vector store")

    with open(vectorizer_path, "wb") as f:
        pickle.dump(vectorizer, f)
