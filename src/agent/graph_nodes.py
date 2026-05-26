from .comparative_retrieval import comparative_retrieval
from .memory import (
    get_session_memory,
    resolve_followup_query,
    update_session_memory,
)
from .router import select_branch
from .state import AgentState
from .topic_extractor import extract_topic_info

from ..core.config import (
    INITIAL_RETRIEVAL_TOP_K,
    RERANK_TOP_K,
)
from ..core.logger import (
    log_query,
    log_retrieval,
)
from ..generation.generator import generate_answer
from ..retrieval.query_rewriter import rewrite_query
from ..retrieval.reranker import rerank
from ..retrieval.retriever import search


def memory_resolution_node(state):
    session_id = state["session_id"]

    memory = get_session_memory(session_id)

    state["memory_history"] = memory.get(
        "history",
        []
    )

    state["current_topic"] = memory.get(
        "current_topic",
        ""
    )

    state["pending_topic"] = memory.get(
        "pending_topic",
        ""
    )

    state["comparison_topics"] = memory.get(
        "comparison_topics",
        []
    )

    resolved_query = resolve_followup_query(
        state["user_query"],
        memory
    )

    state["resolved_query"] = resolved_query

    log_query(
        session_id=state["session_id"],
        user_query=state["user_query"],
        resolved_query=state["resolved_query"],
        branch_taken="pending",
        query_type="pending",
    )

    return state


def routing_node(state):
    agent_state = AgentState(
        user_query=state["user_query"]
    )

    agent_state.resolved_query = state[
        "resolved_query"
    ]

    agent_state.current_topic = state.get(
        "current_topic",
        ""
    )

    agent_state.pending_topic = state.get(
        "pending_topic",
        ""
    )

    branch = select_branch(agent_state)

    state["branch_taken"] = branch
    state["query_type"] = (
        agent_state.query_type
    )

    state["needs_clarification"] = (
        agent_state.needs_clarification
    )

    state["analyzer_confidence"] = (
        agent_state.analyzer_confidence
    )

    state["analyzer_reason"] = (
        agent_state.analyzer_reason
    )

    return state


def topic_extraction_node(state):
    topic_info = extract_topic_info(
        state["resolved_query"],
        branch=state["branch_taken"],
        memory_topic=(
            state.get("pending_topic")
            or state.get("current_topic")
        ),
    )

    state["extracted_topic"] = (
        topic_info.get("main_topic", "")
    )

    state["comparison_topics"] = (
        topic_info.get(
            "comparison_topics",
            []
        )
    )

    return state


def direct_retrieval_node(state):
    retrieval_query = rewrite_query(
        state["resolved_query"]
    )

    state["retrieval_query"] = (
        retrieval_query
    )

    retrieved = search(
        retrieval_query,
        top_k=INITIAL_RETRIEVAL_TOP_K
    )

    reranked = rerank(
        retrieval_query,
        retrieved,
        top_k=RERANK_TOP_K
    )

    state["retrieved_contexts"] = (
        reranked
    )

    log_retrieval(
        session_id=state["session_id"],
        retrieval_query=retrieval_query,
        contexts=reranked,
    )

    return state


def comparative_retrieval_node(state):
    state["retrieval_query"] = (
        state["resolved_query"]
    )

    reranked = comparative_retrieval(
        state["resolved_query"]
    )

    state["retrieved_contexts"] = (
        reranked
    )

    log_retrieval(
        session_id=state["session_id"],
        retrieval_query=state["resolved_query"],
        contexts=reranked,
    )

    return state


def clarification_node(state):
    topic = state.get(
        "extracted_topic",
        ""
    )

    if topic:
        state["answer"] = (
            f"Your query about '{topic}' "
            "is still ambiguous or "
            "underspecified.\n\n"
            "Please provide more detail "
            "or specify the exact "
            "topic/domain."
        )
    else:
        state["answer"] = (
            "Your query appears ambiguous "
            "or underspecified.\n\n"
            "Please provide more detail or "
            "specify the exact topic/domain."
        )

    return state


def out_of_domain_node(state):
    state["answer"] = (
        "This query appears to be outside "
        "the academic knowledge base or "
        "outside the supported scope of "
        "this assistant."
    )

    return state


def generation_node(state):
    answer = generate_answer(
        state["resolved_query"],
        state["retrieved_contexts"]
    )

    state["answer"] = answer

    topic = (
        state.get("extracted_topic")
        or state.get("current_topic")
    )

    if topic:
        state["followup_suggestions"] = [
            (
                f"Compare {topic} with "
                "related concepts"
            ),
            (
                f"What are the applications "
                f"of {topic}?"
            ),
            (
                f"What are the limitations "
                f"of {topic}?"
            ),
        ]

    return state


def memory_update_node(state):
    update_session_memory(
        state["session_id"],
        branch=state["branch_taken"],
        user_query=state["user_query"],
        resolved_query=state[
            "resolved_query"
        ],
        topic=state.get(
            "extracted_topic",
            ""
        ),
        comparison_topics=state.get(
            "comparison_topics",
            []
        ),
        answer=state.get(
            "answer",
            ""
        ),
    )

    refreshed = get_session_memory(
        state["session_id"]
    )

    state["current_topic"] = (
        refreshed.get(
            "current_topic",
            ""
        )
    )

    state["pending_topic"] = (
        refreshed.get(
            "pending_topic",
            ""
        )
    )

    state["memory_history"] = (
        refreshed.get(
            "history",
            []
        )
    )

    return state
