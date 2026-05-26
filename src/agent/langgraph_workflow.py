from langgraph.graph import END
from langgraph.graph import StateGraph

from .graph_nodes import (
    clarification_node,
    comparative_retrieval_node,
    direct_retrieval_node,
    generation_node,
    memory_resolution_node,
    memory_update_node,
    out_of_domain_node,
    routing_node,
    topic_extraction_node,
)
from .graph_state import GraphState
from .state import AgentState


workflow = StateGraph(GraphState)

workflow.add_node(
    "memory_resolution",
    memory_resolution_node
)

workflow.add_node(
    "routing",
    routing_node
)

workflow.add_node(
    "topic_extraction",
    topic_extraction_node
)

workflow.add_node(
    "direct_retrieval",
    direct_retrieval_node
)

workflow.add_node(
    "comparative_retrieval",
    comparative_retrieval_node
)

workflow.add_node(
    "clarification",
    clarification_node
)

workflow.add_node(
    "out_of_domain",
    out_of_domain_node
)

workflow.add_node(
    "generation",
    generation_node
)

workflow.add_node(
    "memory_update",
    memory_update_node
)

workflow.set_entry_point(
    "memory_resolution"
)

workflow.add_edge(
    "memory_resolution",
    "routing"
)

workflow.add_edge(
    "routing",
    "topic_extraction"
)


def route_branch(state):
    branch = state.get(
        "branch_taken",
        "direct_retrieval"
    )

    if branch == "comparative_retrieval":
        return "comparative_retrieval"

    if branch == "clarification_candidate":
        return "clarification"

    if branch == "out_of_domain_response":
        return "out_of_domain"

    return "direct_retrieval"


workflow.add_conditional_edges(
    "topic_extraction",
    route_branch,
    {
        "direct_retrieval": (
            "direct_retrieval"
        ),
        "comparative_retrieval": (
            "comparative_retrieval"
        ),
        "clarification": (
            "clarification"
        ),
        "out_of_domain": (
            "out_of_domain"
        ),
    },
)

workflow.add_edge(
    "direct_retrieval",
    "generation"
)

workflow.add_edge(
    "comparative_retrieval",
    "generation"
)

workflow.add_edge(
    "generation",
    "memory_update"
)

workflow.add_edge(
    "clarification",
    "memory_update"
)

workflow.add_edge(
    "out_of_domain",
    "memory_update"
)

workflow.add_edge(
    "memory_update",
    END
)

app = workflow.compile()


def run_langgraph_workflow(
    user_query: str,
    session_id: str = "default"
) -> AgentState:
    initial_state = {
        "user_query": user_query,
        "resolved_query": "",
        "retrieval_query": "",
        "extracted_topic": "",
        "comparison_topics": [],
        "current_topic": "",
        "pending_topic": "",
        "query_type": "",
        "branch_taken": "",
        "needs_clarification": False,
        "analyzer_confidence": 0.0,
        "analyzer_reason": "",
        "retrieved_contexts": [],
        "answer": "",
        "followup_suggestions": [],
        "session_id": session_id,
        "memory_history": [],
    }

    result = app.invoke(initial_state)

    state = AgentState(
        user_query=result["user_query"]
    )

    for key, value in result.items():
        setattr(state, key, value)

    return state
