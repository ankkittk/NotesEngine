from .query_analyzer import analyze_query
from .state import AgentState


def _has_anchor_topic(state: AgentState) -> bool:
    return bool((state.current_topic or "").strip() or (state.pending_topic or "").strip())


def _normalized(text: str) -> str:
    return " ".join((text or "").strip().split()).lower()


def _is_explicit_comparative(query: str) -> bool:
    q = _normalized(query)

    if q.startswith("compare ") and any(sep in q for sep in (" with ", " and ", " vs ", " versus ", " to ")):
        return True

    if q.startswith("difference between ") and " and " in q:
        return True

    return False


def _is_explicit_direct(query: str) -> bool:
    q = _normalized(query)

    direct_prefixes = (
        "what is ",
        "what are ",
        "explain ",
        "define ",
        "describe ",
        "tell me about ",
        "give me ",
        "write about ",
        "what does ",
        "who is ",
        "how does ",
    )

    return q.startswith(direct_prefixes)


def _is_followup_template(query: str) -> bool:
    q = _normalized(query)

    short_followups = {
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

    first_word = q.split()[0].rstrip("?") if q.split() else ""
    if first_word in short_followups:
        return True

    if q.startswith(("applications of ", "advantages of ", "limitations of ", "benefits of ", "uses of ")):
        return True

    if q.startswith(("in ", "in the ", "for ", "about ", "with ", "on ", "of ")):
        return True

    pronouns = (" it ", " this ", " that ", " them ", " those ", " these ", " its ", " their ")
    if any(p in f" {q} " for p in pronouns):
        return True

    return False


def _branch_to_query_type(branch: str) -> str:
    return {
        "direct_retrieval": "direct",
        "comparative_retrieval": "comparative",
        "clarification_candidate": "ambiguous",
        "out_of_domain_response": "out_of_domain",
    }.get(branch, "direct")


def select_branch(state: AgentState) -> str:
    analysis_query = (state.resolved_query or state.user_query or "").strip()

    analysis = analyze_query(analysis_query)

    state.analyzer_reason = analysis.get("reason", analysis.get("reasoning", ""))
    state.analyzer_confidence = float(analysis.get("confidence", analysis.get("confidence_score", 0.0)) or 0.0)
    state.needs_clarification = bool(analysis.get("needs_clarification", analysis.get("clarification_needed", False)))

    hint = None

    if _is_explicit_comparative(analysis_query):
        hint = "comparative_retrieval"

    elif _is_explicit_direct(analysis_query):
        hint = "direct_retrieval"

    elif _is_followup_template(analysis_query):
        if _has_anchor_topic(state):
            if _normalized(analysis_query).startswith("compare "):
                hint = "comparative_retrieval"
            else:
                hint = "direct_retrieval"

    branch = hint or analysis.get("recommended_branch", "direct_retrieval")

    state.branch_taken = branch
    state.query_type = _branch_to_query_type(branch)

    if hint and hint != analysis.get("recommended_branch"):
        state.analyzer_reason = f"{state.analyzer_reason} (routing overridden by explicit pattern)"

    return branch
