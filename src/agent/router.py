from .query_analyzer import analyze_query
from .state import AgentState


def select_branch(state: AgentState) -> str:
    analysis = analyze_query(state.user_query)

    state.query_type = analysis.get(
        "query_type",
        "direct"
    )

    state.needs_clarification = analysis.get(
        "needs_clarification",
        False
    )

    state.analyzer_confidence = float(
        analysis.get("confidence", 0.0)
    )

    state.analyzer_reason = analysis.get(
        "reason",
        ""
    )

    branch = analysis.get(
        "recommended_branch",
        "direct_retrieval"
    )

    state.branch_taken = branch

    return branch
