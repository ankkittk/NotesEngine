import json
import os
import pickle

import faiss
import numpy as np

from ..core.config import INDEX_PATH, META_PATH, TEXTS_PATH, VECTOR_STORE_PATH, VECTORIZER_PATH


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


def _load_metadata_file():
    if os.path.exists(META_PATH):
        with open(META_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []

    legacy_meta_path = os.path.join(VECTOR_STORE_PATH, "metadata.pkl")
    if os.path.exists(legacy_meta_path):
        with open(legacy_meta_path, "rb") as f:
            data = pickle.load(f)
        return data if isinstance(data, list) else []

    return []


def _save_metadata_file(metadata):
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def store_embeddings(embeddings, texts, metadata, vectorizer):
    os.makedirs(VECTOR_STORE_PATH, exist_ok=True)

    embeddings = np.asarray(embeddings, dtype="float32")
    normalized_metadata = [_normalize_metadata_item(item) for item in metadata]

    if len(texts) != len(normalized_metadata):
        raise ValueError("Texts and metadata length mismatch during vector store write.")

    if os.path.exists(INDEX_PATH):
        index = faiss.read_index(INDEX_PATH)

        existing_texts = []
        if os.path.exists(TEXTS_PATH):
            existing_texts = np.load(TEXTS_PATH, allow_pickle=True).tolist()

        existing_meta = _load_metadata_file()

        existing_meta = [_normalize_metadata_item(item) for item in existing_meta]

        if existing_texts and len(existing_texts) != len(existing_meta):
            raise ValueError("Existing vector store is inconsistent: texts and metadata lengths differ.")

        existing_texts.extend(texts)
        existing_meta.extend(normalized_metadata)

        if len(existing_texts) != len(existing_meta):
            raise ValueError("Appended vector store is inconsistent: texts and metadata lengths differ.")

        index.add(embeddings)

        faiss.write_index(index, INDEX_PATH)
        np.save(TEXTS_PATH, np.array(existing_texts, dtype=object), allow_pickle=True)
        _save_metadata_file(existing_meta)

        print("\nAppended to existing vector store")

    else:
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)

        faiss.write_index(index, INDEX_PATH)
        np.save(TEXTS_PATH, np.array(texts, dtype=object), allow_pickle=True)
        _save_metadata_file(normalized_metadata)

        print("\nCreated new vector store")

    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)
