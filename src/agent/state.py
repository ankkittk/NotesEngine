from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentState:
    user_query: str

    resolved_query: str = ""
    retrieval_query: str = ""

    extracted_topic: str = ""
    comparison_topics: list[str] = field(default_factory=list)

    current_topic: str = ""
    pending_topic: str = ""

    query_type: str = ""
    branch_taken: str = ""

    needs_clarification: bool = False
    analyzer_confidence: float = 0.0
    analyzer_reason: str = ""

    retrieved_contexts: list[dict[str, Any]] = field(default_factory=list)

    answer: str = ""
    followup_suggestions: list[str] = field(default_factory=list)

    session_id: str = "default"
    memory_history: list[dict[str, Any]] = field(default_factory=list)
