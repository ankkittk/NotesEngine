from .query_analyzer import analyze_query
from .state import AgentState


FOLLOWUP_KEYWORDS = {
    "applications",
    "advantages",
    "limitations",
    "benefits",
    "uses",
    "examples",
    "summary",
    "features",
    "purpose",
    "meaning",
}

PRONOUNS = {
    "it",
    "this",
    "that",
    "them",
    "those",
    "these",
    "its",
    "their",
}


def _normalize(text: str) -> str:
    return " ".join(
        (text or "").strip().split()
    ).lower()


def _has_anchor_topic(
    state: AgentState
) -> bool:
    return bool(
        (state.current_topic or "").strip()
        or (state.pending_topic or "").strip()
    )


def _contains_pronoun_reference(
    query: str
) -> bool:
    q = f" {_normalize(query)} "

    return any(
        f" {p} " in q
        for p in PRONOUNS
    )


def _is_short_followup(
    query: str
) -> bool:
    q = _normalize(query).rstrip("?")

    if not q:
        return False

    first_word = (
        q.split()[0]
        if q.split()
        else ""
    )

    return first_word in FOLLOWUP_KEYWORDS


def _is_explicit_comparative(
    query: str
) -> bool:
    q = _normalize(query)

    comparative_patterns = [
        "compare ",
        "difference between ",
    ]

    comparative_separators = [
        " with ",
        " and ",
        " vs ",
        " versus ",
        " to ",
    ]

    return (
        any(
            q.startswith(p)
            for p in comparative_patterns
        )
        and any(
            sep in q
            for sep in comparative_separators
        )
    )


def _is_followup_query(
    query: str
) -> bool:
    q = _normalize(query)

    if _contains_pronoun_reference(q):
        return True

    if _is_short_followup(q):
        return True

    if q.startswith((
        "in ",
        "in the ",
        "for ",
        "about ",
        "on ",
        "with ",
        "of ",
    )):
        return True

    return False


def _query_type_from_branch(
    branch: str
) -> str:
    mapping = {
        "direct_retrieval": "direct",
        "comparative_retrieval": "comparative",
        "clarification_candidate": "ambiguous",
        "out_of_domain_response": "out_of_domain",
    }

    return mapping.get(
        branch,
        "direct"
    )


def _memory_override_branch(
    state: AgentState
) -> str | None:
    resolved_query = (
        state.resolved_query
        or state.user_query
    )

    if not _has_anchor_topic(state):
        return None

    if _is_explicit_comparative(
        resolved_query
    ):
        return (
            "comparative_retrieval"
        )

    if _is_followup_query(
        resolved_query
    ):
        return (
            "direct_retrieval"
        )

    return None


def select_branch(
    state: AgentState
) -> str:
    analysis_query = (
        state.resolved_query
        or state.user_query
        or ""
    ).strip()

    memory_override = (
        _memory_override_branch(
            state
        )
    )

    analysis = analyze_query(
        analysis_query
    )

    state.analyzer_reason = (
        analysis.get(
            "reason",
            analysis.get(
                "reasoning",
                ""
            )
        )
    )

    state.analyzer_confidence = (
        float(
            analysis.get(
                "confidence",
                analysis.get(
                    "confidence_score",
                    0.0
                )
            )
        )
    )

    state.needs_clarification = (
        bool(
            analysis.get(
                "needs_clarification",
                analysis.get(
                    "clarification_needed",
                    False
                )
            )
        )
    )

    branch = (
        analysis.get(
            "recommended_branch",
            "direct_retrieval"
        )
    )

    if memory_override:
        branch = memory_override

        state.analyzer_reason += (
            " | conversational "
            "memory override applied"
        )

    state.branch_taken = branch

    state.query_type = (
        _query_type_from_branch(
            branch
        )
    )

    if (
        branch
        != "clarification_candidate"
    ):
        state.needs_clarification = (
            False
        )

    return branch
