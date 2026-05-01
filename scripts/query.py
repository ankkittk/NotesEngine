import sys
import os

CURRENT_DIR = os.path.dirname(__file__)
SRC_PATH = os.path.abspath(os.path.join(CURRENT_DIR, "..", "src"))
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from retriever import search
from generator import generate_answer
from reranker import rerank


def main():
    while True:
        query = input("\nEnter query (or 'exit'): ").strip()

        if query.lower() == "exit":
            break

        initial_chunks = search(query, top_k=20)
        contexts = rerank(query, initial_chunks, top_k=3)

        print("\n--- Retrieved Context ---\n")
        for i, c in enumerate(contexts, 1):
            print(f"{i}. {c[:200]}...\n")

        answer = generate_answer(query, contexts)

        print("\n--- Final Answer ---\n")
        print(answer)


if __name__ == "__main__":
    main()
