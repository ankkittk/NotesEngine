import os
import sys

CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.core.config import INITIAL_RETRIEVAL_TOP_K, RERANK_TOP_K
from src.generation.generator import generate_answer
from src.retrieval.reranker import rerank
from src.retrieval.retriever import search


def main():
    while True:
        query = input("\nEnter query (or 'exit'): ").strip()

        if query.lower() == "exit":
            break

        initial_chunks = search(query, top_k=INITIAL_RETRIEVAL_TOP_K)
        contexts = rerank(query, initial_chunks, top_k=RERANK_TOP_K)

        print("\n--- Retrieved Context ---\n")
        for i, c in enumerate(contexts, 1):
            print(f"{i}. {c[:200]}...\n")

        answer = generate_answer(query, contexts)

        print("\n--- Final Answer ---\n")
        print(answer)


if __name__ == "__main__":
    main()
