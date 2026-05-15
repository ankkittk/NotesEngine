from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.agent.workflow import run_agentic_workflow

app = FastAPI()


class QueryRequest(BaseModel):
    query: str
    session_id: str = Field(default="default")


@app.get("/")
def root():
    return {"message": "Agentic RAG API running"}


@app.post("/query")
def query(req: QueryRequest):
    state = run_agentic_workflow(req.query, session_id=req.session_id)

    return {
        "query": state.user_query,
        "session_id": state.session_id,
        "query_type": state.query_type,
        "branch_taken": state.branch_taken,
        "needs_clarification": state.needs_clarification,
        "analyzer_confidence": state.analyzer_confidence,
        "analyzer_reason": state.analyzer_reason,
        "resolved_query": state.resolved_query,
        "current_topic": state.current_topic,
        "pending_topic": state.pending_topic,
        "extracted_topic": state.extracted_topic,
        "comparison_topics": state.comparison_topics,
        "retrieval_query": state.retrieval_query,
        "answer": state.answer,
        "contexts": state.retrieved_contexts,
        "followup_suggestions": state.followup_suggestions,
    }
