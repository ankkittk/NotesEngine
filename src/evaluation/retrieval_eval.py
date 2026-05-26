import json
import os
from statistics import mean


QUERY_LOG = "logs/retrieval_logs.jsonl"

OUTPUT_PATH = (
    "eval-results/retrieval_eval.json"
)

os.makedirs(
    "eval-results",
    exist_ok=True
)


def safe_mean(values):
    return round(mean(values), 4) if values else 0.0


def evaluate():
    total_queries = 0
    retrieved_results = 0

    source_frequency = {}

    rerank_scores = []
    retrieval_distances = []

    top1_sources = {}

    if not os.path.exists(QUERY_LOG):
        print("No retrieval logs found.")
        return

    with open(
        QUERY_LOG,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:
            item = json.loads(line)

            total_queries += 1

            results = item.get(
                "results",
                []
            )

            retrieved_results += len(results)

            if results:
                top1 = results[0].get(
                    "source",
                    "unknown"
                )

                top1_sources[top1] = (
                    top1_sources.get(top1, 0)
                    + 1
                )

            for r in results:
                source = r.get(
                    "source",
                    "unknown"
                )

                source_frequency[source] = (
                    source_frequency.get(
                        source,
                        0
                    )
                    + 1
                )

                rerank = r.get(
                    "rerank_score"
                )

                if rerank is not None:
                    rerank_scores.append(
                        float(rerank)
                    )

                distance = r.get(
                    "retrieval_distance"
                )

                if distance is not None:
                    retrieval_distances.append(
                        float(distance)
                    )

    avg_results = (
        retrieved_results / total_queries
        if total_queries
        else 0
    )

    sorted_sources = sorted(
        source_frequency.items(),
        key=lambda x: x[1],
        reverse=True
    )

    top1_distribution = sorted(
        top1_sources.items(),
        key=lambda x: x[1],
        reverse=True
    )

    dominant_source_ratio = 0

    if sorted_sources and retrieved_results:
        dominant_source_ratio = round(
            sorted_sources[0][1]
            / retrieved_results,
            4
        )

    output = {
        "total_queries": total_queries,

        "retrieved_results": (
            retrieved_results
        ),

        "average_results_per_query": (
            round(avg_results, 2)
        ),

        "average_rerank_score": (
            safe_mean(rerank_scores)
        ),

        "average_retrieval_distance": (
            safe_mean(retrieval_distances)
        ),

        "unique_sources_retrieved": (
            len(source_frequency)
        ),

        "dominant_source_ratio": (
            dominant_source_ratio
        ),

        "top_sources": sorted_sources[:10],

        "top1_source_distribution": (
            top1_distribution[:10]
        ),
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
