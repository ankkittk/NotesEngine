import sys
import os

CURRENT_DIR = os.path.dirname(__file__)
SRC_PATH = os.path.abspath(os.path.join(CURRENT_DIR, "..", "src"))
sys.path.append(SRC_PATH)

from retriever import search
from generator import generate_answer


def main():
    while True:
        query = input("\nEnter query (or 'exit'): ")

        if query.lower() == "exit":
            break

        contexts = search(query, top_k=3)

        print("\n--- Retrieved Context ---\n")
        for i, c in enumerate(contexts, 1):
            print(f"{i}. {c[:200]}...\n")

        answer = generate_answer(query, contexts)

        print("\n--- Final Answer ---\n")
        print(answer)


if __name__ == "__main__":
    main()
