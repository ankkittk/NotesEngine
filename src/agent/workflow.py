from .comparative_retrieval import comparative_retrieval
from .memory import get_session_memory, resolve_followup_query, update_session_memory
from .router import select_branch
from .state import AgentState
from .topic_extractor import extract_topic_info

from ..core.config import INITIAL_RETRIEVAL_TOP_K, RERANK_TOP_K
from ..generation.generator import generate_answer
from ..retrieval.query_rewriter import rewrite_query
from ..retrieval.reranker import rerank
from ..retrieval.retriever import search


def _primary_topic(topic: str) -> str:
    topic = " ".join((topic or "").strip().split())

    if not topic:
        return ""

    if " and " in topic and len(topic.split()) > 2:
        return topic.split(" and ", 1)[0].strip()

    return topic


def generate_followup_suggestions(topic: str) -> list[str]:
    topic = _primary_topic(topic)

    if not topic:
        return []

    return [
        f"Compare {topic} with related concepts",
        f"What are the applications of {topic}?",
        f"What are the limitations of {topic}?",
    ]


def handle_out_of_domain(state: AgentState):
    state.answer = (
        "This query appears to be outside the academic knowledge base "
        "or outside the supported scope of this assistant."
    )


def handle_clarification(state: AgentState):
    if state.extracted_topic:
        state.answer = (
            f"Your query about '{state.extracted_topic}' is still ambiguous or underspecified.\n\n"
            "Please provide more detail or specify the exact topic/domain."
        )
        return

    state.answer = (
        "Your query appears ambiguous or underspecified.\n\n"
        "Please provide more detail or specify the exact topic/domain."
    )


def handle_direct_retrieval(state: AgentState):
    retrieval_query = rewrite_query(state.resolved_query)

    state.retrieval_query = retrieval_query

    retrieved = search(
        retrieval_query,
        top_k=INITIAL_RETRIEVAL_TOP_K,
    )

    reranked = rerank(
        retrieval_query,
        retrieved,
        top_k=RERANK_TOP_K,
    )

    state.retrieved_contexts = reranked


def handle_comparative_retrieval(state: AgentState):
    state.retrieval_query = state.resolved_query
    state.retrieved_contexts = comparative_retrieval(state.resolved_query)


def run_agentic_workflow(user_query: str, session_id: str = "default") -> AgentState:
    state = AgentState(
        user_query=user_query,
        session_id=session_id,
    )

    memory = get_session_memory(session_id)

    state.memory_history = list(memory.get("history", []))
    state.current_topic = memory.get("current_topic", "")
    state.pending_topic = memory.get("pending_topic", "")
    state.comparison_topics = list(memory.get("comparison_topics", []))

    resolved_query = resolve_followup_query(user_query, memory)
    state.resolved_query = resolved_query

    branch = select_branch(state)

    topic_info = extract_topic_info(
        state.resolved_query,
        branch=branch,
        memory_topic=state.pending_topic or state.current_topic,
    )

    state.extracted_topic = topic_info.get("main_topic", "")
    state.comparison_topics = topic_info.get("comparison_topics", [])

    if branch == "out_of_domain_response":
        handle_out_of_domain(state)

        update_session_memory(
            session_id,
            branch=branch,
            user_query=user_query,
            resolved_query=state.resolved_query,
            topic="",
            comparison_topics=[],
            answer=state.answer,
        )

        refreshed = get_session_memory(session_id)
        state.current_topic = refreshed.get("current_topic", "")
        state.pending_topic = refreshed.get("pending_topic", "")
        state.memory_history = list(refreshed.get("history", []))

        return state

    if branch == "clarification_candidate":
        handle_clarification(state)

        update_session_memory(
            session_id,
            branch=branch,
            user_query=user_query,
            resolved_query=state.resolved_query,
            topic=state.extracted_topic,
            comparison_topics=state.comparison_topics,
            answer=state.answer,
        )

        refreshed = get_session_memory(session_id)
        state.current_topic = refreshed.get("current_topic", "")
        state.pending_topic = refreshed.get("pending_topic", "")
        state.memory_history = list(refreshed.get("history", []))

        return state

    if branch == "comparative_retrieval":
        handle_comparative_retrieval(state)
    else:
        handle_direct_retrieval(state)

    answer = generate_answer(
        state.resolved_query,
        state.retrieved_contexts,
    )

    state.answer = answer

    topic_for_memory = _primary_topic(
        state.extracted_topic
        or (state.comparison_topics[0] if state.comparison_topics else "")
        or state.current_topic
    )

    state.followup_suggestions = generate_followup_suggestions(topic_for_memory)

    update_session_memory(
        session_id,
        branch=branch,
        user_query=user_query,
        resolved_query=state.resolved_query,
        topic=topic_for_memory,
        comparison_topics=state.comparison_topics,
        answer=answer,
    )

    refreshed = get_session_memory(session_id)
    state.current_topic = refreshed.get("current_topic", "")
    state.pending_topic = refreshed.get("pending_topic", "")
    state.memory_history = list(refreshed.get("history", []))

    return state
