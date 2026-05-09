from sentence_transformers import CrossEncoder

from ..core.config import RERANKER_MODEL_NAME, RERANK_TOP_K

model = CrossEncoder(RERANKER_MODEL_NAME)


def rerank(query, chunks, top_k=RERANK_TOP_K):
    if not chunks:
        return []

    pairs = [(query, chunk) for chunk in chunks]
    scores = model.predict(pairs)

    ranked = sorted(
        zip(chunks, scores),
        key=lambda x: x[1],
        reverse=True
    )

    return [chunk for chunk, _ in ranked[:top_k]]
