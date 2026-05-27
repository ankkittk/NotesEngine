import json
import os
import re
from statistics import mean


GENERATION_LOG = (
    "logs/generation_logs.jsonl"
)

OUTPUT_PATH = (
    "eval-results/grounding_eval.json"
)

os.makedirs(
    "eval-results",
    exist_ok=True
)


def tokenize(text):
    return set(
        re.findall(
            r"\b[a-zA-Z0-9]+\b",
            text.lower()
        )
    )


def sentence_split(text):
    sentences = re.split(
        r"[.!?]\s+",
        text
    )

    return [
        s.strip()
        for s in sentences
        if s.strip()
    ]


def overlap_ratio(a, b):
    if not a:
        return 0.0

    return len(a.intersection(b)) / len(a)


def evaluate():
    if not os.path.exists(
        GENERATION_LOG
    ):
        print(
            "No generation logs found."
        )
        return

    total_answers = 0

    grounding_scores = []

    unsupported_sentences = 0
    total_sentences = 0

    citation_density_scores = []

    with open(
        GENERATION_LOG,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:
            item = json.loads(line)

            total_answers += 1

            answer = item.get(
                "answer",
                ""
            )

            contexts = item.get(
                "contexts",
                []
            )

            combined_context = " ".join(
                c.get("text", "")
                for c in contexts
            )

            answer_tokens = tokenize(
                answer
            )

            context_tokens = tokenize(
                combined_context
            )

            grounding = overlap_ratio(
                answer_tokens,
                context_tokens
            )

            grounding_scores.append(
                grounding
            )

            sentences = sentence_split(
                answer
            )

            total_sentences += len(
                sentences
            )

            for sentence in sentences:
                s_tokens = tokenize(
                    sentence
                )

                s_overlap = overlap_ratio(
                    s_tokens,
                    context_tokens
                )

                if s_overlap < 0.15:
                    unsupported_sentences += 1

            citation_density_scores.append(
                len(contexts)
                / max(1, len(sentences))
            )

    avg_grounding = round(
        mean(grounding_scores),
        4
    ) if grounding_scores else 0

    unsupported_ratio = round(
        unsupported_sentences
        / max(1, total_sentences),
        4
    )

    avg_citation_density = round(
        mean(citation_density_scores),
        4
    ) if citation_density_scores else 0

    output = {
        "total_answers": total_answers,

        "average_grounding_score": (
            avg_grounding
        ),

        "unsupported_sentence_ratio": (
            unsupported_ratio
        ),

        "average_citation_density": (
            avg_citation_density
        ),

        "grounding_quality": (
            "good"
            if avg_grounding >= 0.5
            else "moderate"
            if avg_grounding >= 0.3
            else "weak"
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
