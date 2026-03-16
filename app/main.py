import json
import uuid

from fastapi import FastAPI, File, HTTPException, Response, UploadFile
from fastapi.responses import StreamingResponse

from app.config import settings
from app.models import (
    AskRequest,
    AskResponse,
    EvaluateRequest,
    EvaluateResponse,
    SessionHistoryResponse,
    UploadResponse,
)
from app.services.chunker import chunk_pages
from app.services.document_store import ConversationStore, DocumentMeta, InMemoryDocumentStore
from app.services.ocr_service import GeminiOCRService
from app.services.pdf_parser import extract_pdf_text
from app.services.qa_service import QAService
from app.services.vector_db import ChromaVectorDB


app = FastAPI(
    title="Document Intelligence AI System",
    version="2.0.0",
    description="Upload PDFs and ask questions using an agentic RAG pipeline with OCR, history, and streaming.",
)

store = InMemoryDocumentStore()
conversation_store = ConversationStore(max_turns=settings.history_max_turns)
vector_db = ChromaVectorDB()
qa_service = QAService(vector_db=vector_db)
ocr_service = GeminiOCRService()


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "indexed_chunks": vector_db.collection.count()}


@app.get("/")
def root() -> dict:
    return {
        "message": "Document Intelligence AI System is running.",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    return Response(status_code=204)


@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)) -> UploadResponse:
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    file_bytes = await file.read()

    pages = extract_pdf_text(file_bytes)
    used_ocr = False

    if not pages:
        used_ocr = True
        pages = extract_pdf_text(
            file_bytes,
            ocr_page_fn=lambda page_num: ocr_service.extract_text_for_page(file_bytes, page_num),
        )

    if not pages:
        raise HTTPException(status_code=400, detail="No extractable text found in the PDF, even after OCR.")

    chunks = chunk_pages(
        pages=pages,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    if not chunks:
        raise HTTPException(status_code=400, detail="Unable to create chunks from PDF text.")

    document_id = str(uuid.uuid4())
    qa_service.index_document(document_id=document_id, filename=file.filename, chunks=chunks)
    store.upsert(
        DocumentMeta(
            document_id=document_id,
            filename=file.filename,
            pages=len(pages),
            chunks=len(chunks),
        )
    )

    return UploadResponse(
        document_id=document_id,
        filename=file.filename,
        pages=len(pages),
        chunks=len(chunks),
        used_ocr=used_ocr,
    )


@app.post("/ask", response_model=AskResponse)
def ask_question(payload: AskRequest) -> AskResponse:
    if not vector_db.has_any():
        raise HTTPException(status_code=400, detail="No documents indexed yet. Upload a PDF first.")

    session_id = payload.session_id or str(uuid.uuid4())
    history = conversation_store.get_history(session_id)

    answer, confidence, citations, trace, context, model_used = qa_service.run_agent(
        question=payload.question,
        top_k=payload.top_k,
        document_id=payload.document_id,
        history=history,
        model_profile=payload.model_profile,
        generation_model=payload.generation_model,
    )
    structured_answer = qa_service.to_structured_answer(
        question=payload.question,
        answer=answer,
        context=context,
        model_profile=payload.model_profile,
        generation_model=payload.generation_model,
    )

    conversation_store.append_turn(session_id, "user", payload.question)
    conversation_store.append_turn(session_id, "assistant", answer)

    return AskResponse(
        answer=answer,
        structured_answer=structured_answer,
        confidence=round(confidence, 4),
        citations=citations,
        session_id=session_id,
        agent_trace=trace,
        model_used=model_used,
        model_profile=payload.model_profile,
    )


@app.post("/ask/stream")
def ask_question_stream(payload: AskRequest) -> StreamingResponse:
    if not vector_db.has_any():
        raise HTTPException(status_code=400, detail="No documents indexed yet. Upload a PDF first.")

    session_id = payload.session_id or str(uuid.uuid4())
    history = conversation_store.get_history(session_id)

    tuned_top_k, trace = qa_service.agent.plan(payload.question, payload.top_k)
    citations, confidence, context = qa_service.retrieve(
        question=payload.question,
        top_k=tuned_top_k,
        document_id=payload.document_id,
    )
    trace.append(f"retriever: fetched {len(citations)} chunks")

    def event_stream():
        meta = {
            "type": "meta",
            "session_id": session_id,
            "confidence": round(confidence, 4),
            "citations": [c.model_dump() for c in citations],
            "agent_trace": trace,
            "model_profile": payload.model_profile,
        }
        yield f"data: {json.dumps(meta)}\n\n"

        if not citations:
            fallback = "I could not find enough context in the indexed documents to answer this."
            conversation_store.append_turn(session_id, "user", payload.question)
            conversation_store.append_turn(session_id, "assistant", fallback)
            yield f"data: {json.dumps({'type': 'chunk', 'text': fallback})}\n\n"
            yield 'data: {"type":"done"}\n\n'
            return

        full_text = ""
        token_stream, model_used = qa_service.stream_answer(
            payload.question,
            context,
            history,
            model_profile=payload.model_profile,
            generation_model=payload.generation_model,
        )
        yield f"data: {json.dumps({'type': 'model', 'model_used': model_used})}\n\n"

        for token in token_stream:
            full_text += token
            yield f"data: {json.dumps({'type': 'chunk', 'text': token})}\n\n"

        conversation_store.append_turn(session_id, "user", payload.question)
        conversation_store.append_turn(session_id, "assistant", full_text.strip())
        yield 'data: {"type":"done"}\n\n'

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/evaluate", response_model=EvaluateResponse)
def evaluate(payload: EvaluateRequest) -> EvaluateResponse:
    if not vector_db.has_any():
        raise HTTPException(status_code=400, detail="No documents indexed yet. Upload a PDF first.")

    answer, confidence, _, _, _, _ = qa_service.run_agent(
        question=payload.question,
        top_k=payload.top_k,
        document_id=payload.document_id,
        history=[],
    )
    lexical_f1, semantic_similarity = qa_service.evaluate_answer(
        generated_answer=answer,
        reference_answer=payload.reference_answer,
    )

    return EvaluateResponse(
        generated_answer=answer,
        lexical_f1=round(lexical_f1, 4),
        semantic_similarity=round(semantic_similarity, 4),
        grounded_confidence=round(confidence, 4),
    )


@app.get("/sessions/{session_id}", response_model=SessionHistoryResponse)
def get_session_history(session_id: str) -> SessionHistoryResponse:
    turns = conversation_store.get_history(session_id)
    return SessionHistoryResponse(session_id=session_id, turns=turns)
