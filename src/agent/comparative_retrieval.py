import re

from ..core.config import (
    INITIAL_RETRIEVAL_TOP_K,
    RERANK_TOP_K,
)
from ..retrieval.query_rewriter import rewrite_query
from ..retrieval.reranker import rerank
from ..retrieval.retriever import search


def extract_comparison_targets(query: str) -> list[str]:
    query = (query or "").strip()

    if not query:
        return []

    query = re.sub(
        r"(?i)\bcompare\b",
        "",
        query
    )

    query = re.sub(
        r"(?i)\bdifference between\b",
        "",
        query
    )

    separators = [
        " and ",
        " vs ",
        " versus ",
    ]

    lowered = query.lower()

    for sep in separators:
        if sep in lowered:
            parts = re.split(
                sep,
                query,
                flags=re.IGNORECASE
            )

            cleaned = [
                p.strip(" ?.,")
                for p in parts
                if p.strip()
            ]

            return cleaned

    return [query.strip()]


def deduplicate_chunks(
    chunks: list[dict]
) -> list[dict]:
    seen = set()
    unique_chunks = []

    for chunk in chunks:
        source = chunk.get("source", "")
        page = chunk.get("page", -1)
        text = chunk.get("text", "")

        key = (
            source,
            page,
            text[:200]
        )

        if key in seen:
            continue

        seen.add(key)
        unique_chunks.append(chunk)

    return unique_chunks


def comparative_retrieval(
    query: str
) -> list[dict]:
    targets = extract_comparison_targets(query)

    all_chunks = []

    for target in targets:
        rewritten = rewrite_query(target)

        retrieved = search(
            rewritten,
            top_k=INITIAL_RETRIEVAL_TOP_K
        )

        all_chunks.extend(retrieved)

    unique_chunks = deduplicate_chunks(all_chunks)

    reranked = rerank(
        query,
        unique_chunks,
        top_k=RERANK_TOP_K
    )

    return reranked
