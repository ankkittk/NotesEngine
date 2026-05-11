from sentence_transformers import CrossEncoder

from ..core.config import RERANKER_MODEL_NAME, RERANK_TOP_K
from ..core.utils import get_context_text

model = CrossEncoder(RERANKER_MODEL_NAME)


def rerank(query, chunks, top_k=RERANK_TOP_K):
    if not chunks:
        return []

    pairs = [(query, get_context_text(chunk)) for chunk in chunks]
    scores = model.predict(pairs)

    ranked = sorted(
        zip(chunks, scores),
        key=lambda x: x[1],
        reverse=True
    )

    results = []
    for chunk, score in ranked[:top_k]:
        if isinstance(chunk, dict):
            item = dict(chunk)
        else:
            item = {"text": str(chunk)}
        item["rerank_score"] = float(score)
        results.append(item)

    return results
