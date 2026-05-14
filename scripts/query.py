import os
import sys

CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.agent.workflow import run_agentic_workflow
from src.core.utils import (
    format_citation,
    get_context_text,
    unique_citations,
)


def main():
    while True:
        query = input("\nEnter query (or 'exit'): ").strip()

        if query.lower() == "exit":
            break

        state = run_agentic_workflow(query)

        print("\n--- Agent State ---\n")

        print(f"Query Type            : {state.query_type}")
        print(f"Branch Taken          : {state.branch_taken}")
        print(f"Needs Clarification   : {state.needs_clarification}")
        print(f"Analyzer Confidence   : {state.analyzer_confidence:.2f}")

        if state.analyzer_reason:
            print(f"Analyzer Reason       : {state.analyzer_reason}")

        if state.retrieval_query:
            print(f"Retrieval Query       : {state.retrieval_query}")

        if state.retrieved_contexts:
            print("\n--- Retrieved Context ---\n")

            for i, c in enumerate(state.retrieved_contexts, 1):
                label = format_citation(c)
                snippet = get_context_text(c)[:200]

                extra_bits = []

                if c.get("retrieval_distance") is not None:
                    extra_bits.append(
                        f"faiss distance={c['retrieval_distance']:.4f}"
                    )

                if c.get("rerank_score") is not None:
                    extra_bits.append(
                        f"rerank={c['rerank_score']:.4f}"
                    )

                extra = (
                    f" | {' | '.join(extra_bits)}"
                    if extra_bits else ""
                )

                print(
                    f"{i}. [{label}{extra}] "
                    f"{snippet}...\n"
                )

        print("\n--- Final Answer ---\n")
        print(state.answer)

        if state.retrieved_contexts:
            print("\n--- Sources ---\n")

            for label in unique_citations(
                state.retrieved_contexts
            ):
                print(label)

        if state.followup_suggestions:
            print("\n--- Suggested Follow-ups ---\n")

            for suggestion in state.followup_suggestions:
                print(f"- {suggestion}")


if __name__ == "__main__":
    main()
