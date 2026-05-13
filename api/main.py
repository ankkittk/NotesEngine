from fastapi import FastAPI
from pydantic import BaseModel

from src.agent.workflow import run_agentic_workflow

app = FastAPI()


class QueryRequest(BaseModel):
    query: str


@app.get("/")
def root():
    return {
        "message": "Agentic RAG API running"
    }


@app.post("/query")
def query(req: QueryRequest):
    state = run_agentic_workflow(req.query)

    return {
        "query": state.user_query,
        "query_type": state.query_type,
        "branch_taken": state.branch_taken,
        "retrieval_query": state.retrieval_query,
        "answer": state.answer,
        "contexts": state.retrieved_contexts,
        "followup_suggestions": state.followup_suggestions,
    }
