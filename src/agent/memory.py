from collections import defaultdict
import re


SESSION_MEMORY = defaultdict(
    lambda: {
        "history": [],
        "current_topic": "",
        "pending_topic": "",
        "last_branch": "",
        "last_user_query": "",
        "last_resolved_query": "",
        "comparison_topics": [],
    }
)

_PRONOUNS = {
    "it",
    "this",
    "that",
    "them",
    "those",
    "these",
    "its",
    "their",
}

_SHORT_FOLLOWUPS = {
    "applications",
    "advantages",
    "limitations",
    "benefits",
    "uses",
    "disadvantages",
    "examples",
    "summary",
    "meaning",
    "purpose",
    "features",
}


def _normalize(text: str) -> str:
    return " ".join((text or "").strip().split())


def get_session_memory(session_id: str) -> dict:
    return SESSION_MEMORY[session_id]


def _has_pronoun_reference(query: str) -> bool:
    lower = _normalize(query).lower()
    return any(re.search(rf"\b{p}\b", lower) for p in _PRONOUNS)


def _is_followup_like(query: str) -> bool:
    lower = _normalize(query).lower().rstrip("?")

    if not lower:
        return False

    if _has_pronoun_reference(lower):
        return True

    if lower in _SHORT_FOLLOWUPS:
        return True

    if lower.startswith(("in ", "in the ", "for ", "about ", "with ", "on ", "of ")):
        return True

    return False


def resolve_followup_query(query: str, memory: dict) -> str:
    query = _normalize(query)

    if not query:
        return query

    last_branch = memory.get("last_branch", "")
    current_topic = _normalize(memory.get("current_topic", ""))
    pending_topic = _normalize(memory.get("pending_topic", ""))

    anchor_topic = pending_topic if last_branch == "clarification_candidate" and pending_topic else current_topic or pending_topic

    if not anchor_topic:
        return query

    if last_branch == "out_of_domain_response" and _is_followup_like(query):
        return query

    lower = query.lower()

    if any(re.search(rf"\b{p}\b", lower) for p in _PRONOUNS):
        resolved = query
        for pronoun in _PRONOUNS:
            resolved = re.sub(rf"\b{pronoun}\b", anchor_topic, resolved, flags=re.I)
        query = _normalize(resolved)
        lower = query.lower()

    first_word = lower.split()[0].rstrip("?") if lower.split() else ""

    if first_word in _SHORT_FOLLOWUPS:
        return f"{first_word} of {anchor_topic}"

    if lower.startswith(("in ", "in the ", "for ", "about ", "with ", "on ", "of ")):
        return f"{anchor_topic} {query}"

    return query


def update_session_memory(
    session_id: str,
    *,
    branch: str,
    user_query: str,
    resolved_query: str,
    topic: str = "",
    answer: str = "",
    comparison_topics: list[str] | None = None,
):
    memory = SESSION_MEMORY[session_id]

    topic = _normalize(topic)
    comparison_topics = [_normalize(t) for t in (comparison_topics or []) if _normalize(t)]

    memory["last_user_query"] = user_query
    memory["last_resolved_query"] = resolved_query
    memory["last_branch"] = branch

    memory["history"].append(
        {
            "user_query": user_query,
            "resolved_query": resolved_query,
            "branch": branch,
            "topic": topic,
            "comparison_topics": comparison_topics,
            "answer": answer,
        }
    )

    if branch == "clarification_candidate":
        if topic:
            memory["pending_topic"] = topic
        memory["comparison_topics"] = comparison_topics
        return

    if branch == "out_of_domain_response":
        memory["pending_topic"] = ""
        memory["comparison_topics"] = []
        return

    if topic:
        memory["current_topic"] = topic
    elif comparison_topics:
        memory["current_topic"] = comparison_topics[0]

    memory["pending_topic"] = ""
    memory["comparison_topics"] = comparison_topics
