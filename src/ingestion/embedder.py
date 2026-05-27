from sentence_transformers import SentenceTransformer
import numpy as np

from ..core.config import EMBEDDING_MODEL_NAME

_model = None


def get_embedding_model():
    global _model

    if _model is None:
        _model = SentenceTransformer(
            EMBEDDING_MODEL_NAME
        )

    return _model


def create_embeddings(chunks):
    texts = [c["text"] for c in chunks]

    metadata = [
        {
            "source": c.get("source", "unknown"),
            "page": c.get("page"),
            "chunk_id": c.get("chunk_id"),
        }
        for c in chunks
    ]

    model = get_embedding_model()

    embeddings = model.encode(
        texts,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    ).astype("float32")

    return embeddings, texts, metadata, model
