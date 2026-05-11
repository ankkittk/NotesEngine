from fastapi import FastAPI
from pydantic import BaseModel

from src.retrieval.retriever import search
from src.retrieval.reranker import rerank
from src.generation.generator import generate_answer
from src.core.config import (
    INITIAL_RETRIEVAL_TOP_K,
    RERANK_TOP_K
)

app = FastAPI()


class QueryRequest(BaseModel):
    query: str


@app.get("/")
def root():
    return {"status": "running"}


@app.post("/query")
def query_rag(req: QueryRequest):
    initial_chunks = search(
        req.query,
        top_k=INITIAL_RETRIEVAL_TOP_K
    )

    contexts = rerank(
        req.query,
        initial_chunks,
        top_k=RERANK_TOP_K
    )

    answer = generate_answer(req.query, contexts)

    return {
        "query": req.query,
        "answer": answer,
        "contexts": contexts
    }
