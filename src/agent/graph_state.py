from typing import Any, TypedDict


class GraphState(TypedDict):
    user_query: str

    resolved_query: str
    retrieval_query: str

    extracted_topic: str
    comparison_topics: list[str]

    current_topic: str
    pending_topic: str

    query_type: str
    branch_taken: str

    needs_clarification: bool
    analyzer_confidence: float
    analyzer_reason: str

    retrieved_contexts: list[dict[str, Any]]

    answer: str

    followup_suggestions: list[str]

    session_id: str

    memory_history: list[dict[str, Any]]
