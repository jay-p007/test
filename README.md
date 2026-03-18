# Document Intelligence AI System

A complete implementation for the AI/ML Engineer Machine Test.

## What Is Implemented

### Core Requirements
- PDF upload
- PDF text extraction
- Document question answering
- AI-generated answers

### Optional Features (Now Fully Implemented)
- Retrieval Augmented Generation (RAG)
- Persistent vector database search (ChromaDB)
- Source citations
- Confidence score
- Multi-document support
- Agent-based workflow (planner -> retriever -> writer -> critic)
- Structured API responses
- Evaluation logic endpoint
- Conversation history (session-based)
- Streaming responses (SSE)
- OCR fallback for scanned PDFs (Gemini vision)
- Streamlit UI
- UI model selection (cost profile)
- Rate-limit fallback across model candidates
- Docker support

## Architecture

- **FastAPI backend** with REST + streaming endpoints.
- **PDF parsing layer** with `pypdf` for native text.
- **OCR fallback layer** with `PyMuPDF` rendering + Gemini OCR prompt.
- **Chunking layer** for retrieval-friendly text spans.
- **Embedding layer** using Gemini embeddings.
- **Vector DB layer** using persistent ChromaDB.
- **Agent layer** for query planning and answer refinement.
- **Conversation store** for session history.
- **Evaluation layer** for lexical and semantic quality metrics.

## Project Structure

```text
.
├── app
│   ├── main.py
│   ├── config.py
│   ├── models.py
│   └── services
│       ├── agent.py
│       ├── chunker.py
│       ├── document_store.py
│       ├── ocr_service.py
│       ├── pdf_parser.py
│       ├── qa_service.py
│       ├── vector_db.py
│       └── vector_index.py
├── data
│   └── .gitkeep
├── .env.example
├── .gitignore
├── Dockerfile
├── README.md
├── requirements.txt
└── streamlit_app.py
```

## Setup

1. Create env and install:

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Create `.env`:

```bash
copy .env.example .env
```

3. Set key in `.env`:
- `GEMINI_API_KEY`

## Run

### Backend

```bash
uvicorn app.main:app --reload
```

Open docs:
- `http://127.0.0.1:8000/docs`

### Streamlit UI

```bash
streamlit run streamlit_app.py
```

Open:
- `http://localhost:8501`

## API Endpoints

- `POST /upload`
- `POST /ask`
- `POST /ask/stream` (SSE streaming)
- `POST /evaluate`
- `GET /sessions/{session_id}`
- `GET /health`

## Key Request Examples

### Ask (with history)

```json
{
  "question": "What is the termination clause?",
  "document_id": "optional-doc-id",
  "top_k": 5,
  "session_id": "optional-session-id",
  "model_profile": "low_cost | balanced | high_quality"
}
```

### Evaluate

```json
{
  "question": "What is the termination clause?",
  "reference_answer": "Either party may terminate with 30 days notice.",
  "document_id": "optional-doc-id",
  "top_k": 5
}
```

## Design Decisions

- Persistent ChromaDB chosen over in-memory vectors for better machine-test completeness.
- OCR runs only when normal extraction fails, reducing cost/latency for native-text PDFs.
- Agent workflow is explicit and traceable in output for explainability.
- Conversation history is session-based and bounded (`HISTORY_MAX_TURNS`) to limit prompt growth.
- Evaluation combines lexical overlap (F1) and semantic similarity for practical quality signals.

## Limitations

- OCR quality depends on scan quality and Gemini vision output.
- Conversation and document metadata are in-memory, while vectors persist in Chroma.
- No auth/rate limiting for production.
- Streaming endpoint currently emits SSE text chunks and metadata; no WebSocket transport.

## Docker

```bash
docker build -t doc-intelligence-ai .
docker run --rm -p 8000:8000 --env-file .env doc-intelligence-ai
```

## Deploy On Render

This repo includes `render.yaml` for one-click multi-service deploy (FastAPI backend + Streamlit UI).

1. Push this repo to GitHub.
2. In Render, choose **New +** -> **Blueprint**.
3. Select your repo (Render will detect `render.yaml`).
4. Set secret env var `GEMINI_API_KEY` for `doc-intelligence-backend`.
5. Deploy.

After deploy:
- Backend URL: `https://doc-intelligence-backend.onrender.com`
- UI URL: `https://doc-intelligence-ui.onrender.com`

The UI service reads `BACKEND_URL` automatically from the backend service URL via blueprint config.
