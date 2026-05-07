import os
import numpy as np
import faiss
import pickle

from config import VECTOR_STORE_PATH, INDEX_FILE, TEXTS_FILE, VECTORIZER_FILE

META_FILE = "metadata.pkl"


def store_embeddings(embeddings, texts, metadata, vectorizer):
    os.makedirs(VECTOR_STORE_PATH, exist_ok=True)

    index_path = os.path.join(VECTOR_STORE_PATH, INDEX_FILE)
    texts_path = os.path.join(VECTOR_STORE_PATH, TEXTS_FILE)
    vectorizer_path = os.path.join(VECTOR_STORE_PATH, VECTORIZER_FILE)
    meta_path = os.path.join(VECTOR_STORE_PATH, META_FILE)

    embeddings = np.asarray(embeddings, dtype="float32")

    if os.path.exists(index_path):
        index = faiss.read_index(index_path)

        existing_texts = np.load(texts_path, allow_pickle=True).tolist()
        existing_texts.extend(texts)

        with open(meta_path, "rb") as f:
            existing_meta = pickle.load(f)
        existing_meta.extend(metadata)

        index.add(embeddings)

        faiss.write_index(index, index_path)
        np.save(texts_path, np.array(existing_texts, dtype=object), allow_pickle=True)

        with open(meta_path, "wb") as f:
            pickle.dump(existing_meta, f)

        print("\nAppended to existing vector store")

    else:
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)

        faiss.write_index(index, index_path)
        np.save(texts_path, np.array(texts, dtype=object), allow_pickle=True)

        with open(meta_path, "wb") as f:
            pickle.dump(metadata, f)

        print("\nCreated new vector store")

    with open(vectorizer_path, "wb") as f:
        pickle.dump(vectorizer, f)
