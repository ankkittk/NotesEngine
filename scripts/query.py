import os
import sys

CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.core.config import INITIAL_RETRIEVAL_TOP_K, RERANK_TOP_K
from src.core.utils import format_citation, get_context_text, unique_citations
from src.generation.generator import generate_answer
from src.retrieval.query_rewriter import rewrite_query
from src.retrieval.reranker import rerank
from src.retrieval.retriever import search


def main():
    while True:
        query = input("\nEnter query (or 'exit'): ").strip()

        if query.lower() == "exit":
            break

        retrieval_query = rewrite_query(query)

        if retrieval_query != query:
            print("\n--- Rewritten Retrieval Query ---\n")
            print(retrieval_query)

        initial_chunks = search(retrieval_query, top_k=INITIAL_RETRIEVAL_TOP_K)
        contexts = rerank(retrieval_query, initial_chunks, top_k=RERANK_TOP_K)

        print("\n--- Retrieved Context ---\n")
        for i, c in enumerate(contexts, 1):
            label = format_citation(c)
            snippet = get_context_text(c)[:200]

            extra_bits = []
            if isinstance(c, dict) and c.get("retrieval_distance") is not None:
                extra_bits.append(f"faiss distance={c['retrieval_distance']:.4f}")
            if isinstance(c, dict) and c.get("rerank_score") is not None:
                extra_bits.append(f"rerank={c['rerank_score']:.4f}")

            extra = f" | {' | '.join(extra_bits)}" if extra_bits else ""
            print(f"{i}. [{label}{extra}] {snippet}...\n")

        answer = generate_answer(query, contexts)

        print("\n--- Final Answer ---\n")
        print(answer)

        if contexts:
            print("\n--- Sources ---\n")
            for label in unique_citations(contexts):
                print(label)


if __name__ == "__main__":
    main()
