from .router import select_branch
from .state import AgentState

from ..core.config import INITIAL_RETRIEVAL_TOP_K, RERANK_TOP_K
from ..generation.generator import generate_answer
from ..retrieval.query_rewriter import rewrite_query
from ..retrieval.reranker import rerank
from ..retrieval.retriever import search


def generate_followup_suggestions(query: str) -> list[str]:
    query = query.strip()

    if not query:
        return []

    return [
        f"Compare {query} with related concepts",
        f"What are the applications of {query}?",
        f"What are the limitations of {query}?",
    ]


def run_agentic_workflow(user_query: str) -> AgentState:
    state = AgentState(user_query=user_query)

    branch = select_branch(state)

    retrieval_query = rewrite_query(user_query)

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

    answer = generate_answer(
        user_query,
        reranked
    )

    state.answer = answer

    state.followup_suggestions = generate_followup_suggestions(
        user_query
    )

    return state
