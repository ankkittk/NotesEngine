from .comparative_retrieval import comparative_retrieval
from .router import select_branch
from .state import AgentState

from ..core.config import (
    INITIAL_RETRIEVAL_TOP_K,
    RERANK_TOP_K,
)
from ..generation.generator import generate_answer
from ..retrieval.query_rewriter import rewrite_query
from ..retrieval.reranker import rerank
from ..retrieval.retriever import search


def generate_followup_suggestions(
    query: str
) -> list[str]:
    query = query.strip()

    if not query:
        return []

    return [
        f"Compare '{query}' with related concepts",
        f"What are the applications of '{query}'?",
        f"What are the limitations of '{query}'?",
    ]


def handle_out_of_domain(
    state: AgentState
):
    state.answer = (
        "This query appears to be outside the academic "
        "knowledge base or outside the supported scope "
        "of this assistant."
    )


def handle_clarification(
    state: AgentState
):
    state.answer = (
        "Your query appears ambiguous or underspecified.\n\n"
        "Please provide more detail or specify the exact "
        "topic/domain."
    )


def handle_direct_retrieval(
    state: AgentState
):
    retrieval_query = rewrite_query(
        state.user_query
    )

    state.retrieval_query = retrieval_query

    retrieved = search(
        retrieval_query,
        top_k=INITIAL_RETRIEVAL_TOP_K
    )

    reranked = rerank(
        retrieval_query,
        retrieved,
        top_k=RERANK_TOP_K
    )

    state.retrieved_contexts = reranked


def handle_comparative_retrieval(
    state: AgentState
):
    state.retrieval_query = (
        state.user_query
    )

    reranked = comparative_retrieval(
        state.user_query
    )

    state.retrieved_contexts = reranked


def run_agentic_workflow(
    user_query: str
) -> AgentState:
    state = AgentState(
        user_query=user_query
    )

    branch = select_branch(state)

    if branch == "out_of_domain_response":
        handle_out_of_domain(state)
        return state

    if branch == "clarification_candidate":
        handle_clarification(state)
        return state

    if branch == "comparative_retrieval":
        handle_comparative_retrieval(state)

    else:
        handle_direct_retrieval(state)

    answer = generate_answer(
        state.user_query,
        state.retrieved_contexts
    )

    state.answer = answer

    state.followup_suggestions = (
        generate_followup_suggestions(
            state.user_query
        )
    )

    return state
