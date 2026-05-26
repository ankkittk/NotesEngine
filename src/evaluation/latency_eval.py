import json
import os
from collections import Counter


QUERY_LOG = "logs/queries.jsonl"

OUTPUT_PATH = (
    "eval-results/latency_eval.json"
)

os.makedirs(
    "eval-results",
    exist_ok=True
)


def evaluate():
    if not os.path.exists(QUERY_LOG):
        print("No query logs found.")
        return

    total_queries = 0

    branch_counter = Counter()
    query_type_counter = Counter()
    session_counter = Counter()

    with open(
        QUERY_LOG,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:
            item = json.loads(line)

            total_queries += 1

            branch = item.get(
                "branch_taken",
                "unknown"
            )

            qtype = item.get(
                "query_type",
                "unknown"
            )

            session = item.get(
                "session_id",
                "unknown"
            )

            branch_counter[branch] += 1
            query_type_counter[qtype] += 1
            session_counter[session] += 1

    output = {
        "total_queries": total_queries,

        "unique_sessions": (
            len(session_counter)
        ),

        "average_queries_per_session": round(
            total_queries / max(
                1,
                len(session_counter)
            ),
            2
        ),

        "branch_distribution": dict(
            branch_counter
        ),

        "query_type_distribution": dict(
            query_type_counter
        ),

        "top_sessions": (
            session_counter.most_common(10)
        ),

        "status": "logging active"
    }

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            output,
            f,
            indent=2
        )

    print(
        json.dumps(
            output,
            indent=2
        )
    )


if __name__ == "__main__":
    evaluate()
