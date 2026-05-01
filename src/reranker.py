from sentence_transformers import CrossEncoder

model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def rerank(query, chunks, top_k=3):
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
