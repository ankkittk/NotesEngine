import sys
import os

sys.path.append(os.path.abspath("src"))

from retriever import search


def main():
    while True:
        query = input("\nEnter query (or 'exit'): ")

        if query.lower() == "exit":
            break

        results = search(query, top_k=3)

        print("\n--- Top Results ---\n")
        for i, res in enumerate(results, 1):
            print(f"{i}. {res[:300]}...\n")


if __name__ == "__main__":
    main()
