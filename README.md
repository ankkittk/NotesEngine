# NotesEngine

NotesEngine is a personal Agentic RAG chatbot for study material. It turns PDFs, text files, Word documents, and slide decks into a searchable knowledge base, then answers questions with retrieved context, citations, conversational memory, and an agent workflow that can route between direct lookup, comparison, clarification, and out-of-domain handling.

This is not just a vector search wrapper. NotesEngine has an ingestion pipeline, FAISS-backed retrieval, query rewriting, cross-encoder reranking, LangGraph orchestration, grounded answer generation, Streamlit chat UI, FastAPI backend, telemetry logs, and evaluation scripts for retrieval and grounding quality.

## What It Does

- Ingests `.pdf`, `.txt`, `.docx`, and `.pptx` files into a local knowledge base.
- Extracts text from normal documents and uses a Gemini vision fallback for image-heavy PDF pages.
- Chunks notes with source, page, and chunk metadata.
- Embeds chunks with `all-MiniLM-L6-v2` and stores them in a local FAISS index.
- Rewrites user questions into retrieval-friendly queries while preserving technical terms.
- Retrieves broad candidates, then reranks them with a cross-encoder.
- Uses an agent router to detect direct questions, comparisons, vague prompts, follow-ups, and out-of-domain queries.
- Maintains lightweight session memory so follow-ups like "what about its limitations?" can resolve to the current topic.
- Generates grounded answers that quote relevant note snippets and cite their source labels.
- Logs queries, retrievals, ingestion events, and generations for later inspection.
- Evaluates retrieval distribution, grounding quality, and branch/query behavior from logs.

## Architecture

```text
                 +---------------------------+
                 |  PDFs / TXT / DOCX / PPTX |
                 +-------------+-------------+
                               |
                               v
                 +---------------------------+
                 | Ingestion + Text Loading  |
                 | PyMuPDF / DOCX / PPTX     |
                 | Vision OCR fallback       |
                 +-------------+-------------+
                               |
                               v
                 +---------------------------+
                 | Chunking + Metadata       |
                 | source / page / chunk_id  |
                 +-------------+-------------+
                               |
                               v
                 +---------------------------+
                 | Embeddings + FAISS Store  |
                 | all-MiniLM-L6-v2          |
                 +-------------+-------------+
                               |
                               v
User Query --> Session Memory --> Agent Router --> Retrieval Branch
                               |                    |
                               |                    v
                               |          Query Rewrite + Search
                               |                    |
                               |                    v
                               |          Cross-Encoder Rerank
                               |                    |
                               v                    v
                         Clarify / Reject <-- Grounded Generation
                                                    |
                                                    v
                                           Answer + Citations
                                                    |
                                                    v
                                           Logs + Evaluation
```

## Agent Workflow

NotesEngine uses a LangGraph workflow for the API path:

```text
memory_resolution
      |
      v
routing
      |
      v
topic_extraction
      |
      +--> direct_retrieval ------+
      |                           |
      +--> comparative_retrieval -+--> generation --> memory_update --> END
      |                           |
      +--> clarification ---------+
      |                           |
      +--> out_of_domain ---------+
```

The router decides how to handle the query before retrieval happens:

- `direct_retrieval`: normal academic question answering from notes.
- `comparative_retrieval`: comparison-style prompts such as "compare X and Y".
- `clarification_candidate`: vague prompts that need more detail.
- `out_of_domain_response`: prompts outside the academic knowledge base.

## Tech Stack

- Python
- FastAPI + Uvicorn for the backend API
- Streamlit for the chat interface
- LangGraph for the agent workflow
- Sentence Transformers for embeddings and reranking
- FAISS for local vector search
- PyMuPDF, python-docx, and python-pptx for document loading
- Groq OpenAI-compatible API for LLM generation and analysis
- Gemini API for vision-based OCR fallback
- JSONL logs and local evaluation scripts

## Quick Start

### 1. Create a virtual environment

```powershell
python -m venv venv
.\venv\Scripts\activate
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Configure environment variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
GEMINI_API_KEY=your_gemini_api_key
```

`GROQ_API_KEY` powers query analysis, query rewriting, topic extraction, and answer generation. `GEMINI_API_KEY` is used only when the ingestion pipeline needs vision OCR for image-heavy PDF pages.

### 4. Add documents

Place supported files in the `data/` directory:

```text
data/
  your_notes.pdf
  lecture_slides.pptx
  summary.docx
```

You can also upload documents later through the Streamlit UI or the FastAPI upload endpoint.

### 5. Build the vector store

```powershell
python scripts/run_ingestion.py
```

This extracts text, chunks documents, creates embeddings, and writes the FAISS index into `vector-db/`.

### 6. Run the API

```powershell
uvicorn api.main:app --reload
```

The API runs at:

```text
http://127.0.0.1:8000
```

### 7. Run the chat UI

In another terminal:

```powershell
streamlit run ui/app.py
```

The UI lets you upload notes, start a session, ask questions, inspect retrieved contexts, and see suggested follow-ups.


## Docker

### Build Backend Image

```powershell
docker build -f Dockerfile.backend -t notesengine-backend .
```

### Run Backend Container

```powershell
docker run --env-file .env -p 8000:8000 notesengine-backend
```

The backend will be available at:

```text
http://localhost:8000
```

### Verify API

Open:

```text
http://localhost:8000/docs
```

### Run Frontend

In a separate terminal:

```powershell
streamlit run ui/app.py
```

## Usage

### Streamlit Chat

Use the Streamlit app for the full interactive experience:

```powershell
streamlit run ui/app.py
```

### CLI Chat

Use the terminal query loop:

```powershell
python scripts/query.py
```

The CLI prints the agent state, branch decision, retrieved contexts, final answer, citations, and follow-up suggestions.

### FastAPI Query

```powershell
curl -X POST http://127.0.0.1:8000/query `
  -H "Content-Type: application/json" `
  -d "{\"query\":\"What are the applications of logistic regression?\",\"session_id\":\"demo-session\"}"
```

### FastAPI Upload

```powershell
curl -X POST http://127.0.0.1:8000/upload `
  -F "session_id=demo-session" `
  -F "files=@data/your_notes.pdf"
```

## API Reference

### `GET /`

Health check endpoint.

```json
{
  "message": "LangGraph Agentic RAG API running"
}
```

### `POST /query`

Runs the LangGraph RAG workflow for a user query.

Request:

```json
{
  "query": "Compare Naive Bayes with Bayesian Networks",
  "session_id": "demo-session"
}
```

Response includes:

- `query`: original user query.
- `session_id`: conversation/session identifier.
- `query_type`: direct, comparative, ambiguous, or out-of-domain classification.
- `branch_taken`: workflow branch selected by the router.
- `resolved_query`: memory-aware query after follow-up resolution.
- `current_topic`, `pending_topic`, `extracted_topic`: topic tracking fields.
- `comparison_topics`: topics detected for comparison prompts.
- `answer`: final grounded response.
- `contexts`: reranked source chunks used for generation.
- `followup_suggestions`: suggested next questions.

### `POST /upload`

Uploads and ingests one or more documents.

Form fields:

- `files`: one or more `.pdf`, `.txt`, `.docx`, or `.pptx` files.
- `session_id`: optional session identifier.

Response includes saved files, ingestion results, and skipped files with reasons.

## Evaluation

NotesEngine logs runtime behavior into JSONL files under `logs/`, then turns those logs into compact evaluation summaries under `eval-results/`.

Run retrieval evaluation:

```powershell
python src/evaluation/retrieval_eval.py
```

Run grounding evaluation:

```powershell
python src/evaluation/grounding_eval.py
```

Run query and branch distribution evaluation:

```powershell
python src/evaluation/latency_eval.py
```

Current local sample results:

| Area | Signal | Sample result |
| --- | --- | --- |
| Retrieval | Logged retrieval queries | 17 |
| Retrieval | Average returned contexts per query | 3.0 |
| Grounding | Average grounding score | 0.6157 |
| Grounding | Unsupported sentence ratio | 0.0 |
| Grounding | Grounding quality | good |
| Routing | Logged API/UI queries | 6 |
| Routing | Branches observed | direct retrieval, comparative retrieval |

These numbers are local development snapshots from existing logs, not benchmark claims.

## Project Structure

```text
NotesEngine/
  api/
    main.py                  FastAPI routes for query and upload
  ui/
    app.py                   Streamlit chat interface
  scripts/
    run_ingestion.py         Batch ingestion from data/
    query.py                 CLI chat loop
  src/
    agent/                   Router, memory, LangGraph workflow, comparison logic
    core/                    Config, logging, shared utilities
    evaluation/              Retrieval, grounding, and query-log evaluation
    generation/              Prompting and answer generation
    ingestion/               Loaders, chunking, embeddings, vision OCR fallback
    retrieval/               FAISS search, query rewriting, reranking
  data/                      Local source documents
  vector-db/                 Local FAISS index and metadata
  logs/                      JSONL telemetry logs
  eval-results/              Evaluation outputs
  Dockerfile.backend         Backend container definition
  Dockerfile.frontend        Frontend container definition
  docker-compose.yml         Local multi-container setup
  .dockerignore              Docker build exclusions
```

## Design Notes

NotesEngine is built around one principle: answers should stay close to the notes. Retrieval brings back source chunks with page metadata, reranking narrows the context, and generation is prompted to quote relevant lines before explaining. When the system does not have enough information, it should clarify, stay partial, or reject the query instead of inventing unsupported details.

The result is a study assistant that behaves less like a generic chatbot and more like a careful reader sitting on top of your own material.
