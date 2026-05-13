from .state import AgentState


AMBIGUOUS_SHORT_QUERIES = {
    "svm",
    "cnn",
    "rnn",
    "nlp",
    "natural acceptance",
    "bayes",
    "regression",
}


def detect_query_type(query: str) -> str:
    query = (query or "").strip().lower()

    if not query:
        return "empty"

    if query in AMBIGUOUS_SHORT_QUERIES:
        return "ambiguous"

    if len(query.split()) <= 2:
        return "short_query"

    comparative_keywords = [
        "compare",
        "difference",
        "vs",
        "versus",
    ]

    if any(word in query for word in comparative_keywords):
        return "comparative"

    return "direct"


def select_branch(state: AgentState) -> str:
    qtype = detect_query_type(state.user_query)

    state.query_type = qtype

    if qtype == "ambiguous":
        state.branch_taken = "clarification_candidate"
        return "clarification_candidate"

    if qtype == "comparative":
        state.branch_taken = "comparative_retrieval"
        return "comparative_retrieval"

    state.branch_taken = "direct_retrieval"
    return "direct_retrieval"
