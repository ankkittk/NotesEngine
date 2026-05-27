import json
import os
from datetime import datetime


LOG_DIR = "logs"

QUERY_LOG = os.path.join(
    LOG_DIR,
    "queries.jsonl"
)

RETRIEVAL_LOG = os.path.join(
    LOG_DIR,
    "retrieval_logs.jsonl"
)

INGESTION_LOG = os.path.join(
    LOG_DIR,
    "ingestion_logs.jsonl"
)

GENERATION_LOG = os.path.join(
    LOG_DIR,
    "generation_logs.jsonl"
)

ERROR_LOG = os.path.join(
    LOG_DIR,
    "errors.log"
)


os.makedirs(LOG_DIR, exist_ok=True)


def _timestamp():
    return datetime.utcnow().isoformat()


def _append_jsonl(path, payload):
    with open(path, "a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                payload,
                ensure_ascii=False
            )
            + "\n"
        )


def log_query(
    session_id,
    user_query,
    resolved_query,
    branch_taken,
    query_type,
):
    payload = {
        "timestamp": _timestamp(),
        "session_id": session_id,
        "user_query": user_query,
        "resolved_query": resolved_query,
        "branch_taken": branch_taken,
        "query_type": query_type,
    }

    _append_jsonl(
        QUERY_LOG,
        payload
    )


def log_retrieval(
    session_id,
    retrieval_query,
    contexts,
):
    payload = {
        "timestamp": _timestamp(),
        "session_id": session_id,
        "retrieval_query": retrieval_query,
        "results": [],
    }

    for c in contexts:
        payload["results"].append({
            "source": c.get("source"),
            "page": c.get("page"),
            "chunk_id": c.get("chunk_id"),
            "retrieval_distance": c.get(
                "retrieval_distance"
            ),
            "rerank_score": c.get(
                "rerank_score"
            ),
        })

    _append_jsonl(
        RETRIEVAL_LOG,
        payload
    )


def log_generation(
    session_id,
    query,
    answer,
    contexts,
):
    payload = {
        "timestamp": _timestamp(),
        "session_id": session_id,
        "query": query,
        "answer": answer,
        "contexts": [],
    }

    for c in contexts:
        payload["contexts"].append({
            "source": c.get("source"),
            "page": c.get("page"),
            "text": c.get("text", ""),
        })

    _append_jsonl(
        GENERATION_LOG,
        payload
    )


def log_ingestion(result):
    payload = {
        "timestamp": _timestamp(),
        **result
    }

    _append_jsonl(
        INGESTION_LOG,
        payload
    )


def log_error(message):
    with open(
        ERROR_LOG,
        "a",
        encoding="utf-8"
    ) as f:
        f.write(
            f"[{_timestamp()}] {message}\n"
        )
