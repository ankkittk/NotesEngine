import os
import shutil
import uuid
from pathlib import Path

from fastapi.middleware.cors import CORSMiddleware

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from src.agent.langgraph_workflow import run_langgraph_workflow
from src.core.config import ALLOWED_EXTENSIONS, DATA_PATH
from src.ingestion.ingest_service import ingest_multiple_files
from src.ingestion.ingestion_tracker import load_tracker

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str
    session_id: str = Field(default="default")


def _ensure_data_dir():
    os.makedirs(DATA_PATH, exist_ok=True)


def _is_allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def _unique_destination_path(original_name: str, processed_files: set[str]) -> str:
    _ensure_data_dir()

    base = Path(original_name).stem
    suffix = Path(original_name).suffix
    candidate = Path(DATA_PATH) / original_name

    if not candidate.exists() and original_name not in processed_files:
        return str(candidate)

    unique_name = f"{base}_{uuid.uuid4().hex[:8]}{suffix}"
    return str(Path(DATA_PATH) / unique_name)


def _save_upload_file(upload: UploadFile, processed_files: set[str]) -> str:
    destination = _unique_destination_path(upload.filename, processed_files)

    with open(destination, "wb") as buffer:
        shutil.copyfileobj(upload.file, buffer)

    return destination


@app.get("/")
def root():
    return {"message": "LangGraph Agentic RAG API running"}


@app.post("/query")
def query(req: QueryRequest):
    state = run_langgraph_workflow(req.query, session_id=req.session_id)

    return {
        "query": state.user_query,
        "session_id": state.session_id,
        "query_type": state.query_type,
        "branch_taken": state.branch_taken,
        "resolved_query": state.resolved_query,
        "current_topic": state.current_topic,
        "pending_topic": state.pending_topic,
        "extracted_topic": state.extracted_topic,
        "comparison_topics": state.comparison_topics,
        "answer": state.answer,
        "contexts": state.retrieved_contexts,
        "followup_suggestions": state.followup_suggestions,
    }


@app.post("/upload")
def upload_files(
    files: list[UploadFile] = File(...),
    session_id: str = Form(default="default"),
):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    processed_files = load_tracker()
    saved_paths = []
    skipped = []

    for upload in files:
        if not upload.filename:
            skipped.append(
                {
                    "file_name": "",
                    "status": "error",
                    "reason": "missing_filename",
                }
            )
            continue

        if not _is_allowed_file(upload.filename):
            skipped.append(
                {
                    "file_name": upload.filename,
                    "status": "error",
                    "reason": "unsupported_file_type",
                }
            )
            continue

        saved_path = _save_upload_file(upload, processed_files)
        saved_paths.append(saved_path)

    results = ingest_multiple_files(saved_paths) if saved_paths else []

    return {
        "session_id": session_id,
        "saved_files": [os.path.basename(p) for p in saved_paths],
        "results": results,
        "skipped": skipped,
    }
