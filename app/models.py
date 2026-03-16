from typing import List, Optional

from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    document_id: str
    filename: str
    pages: int
    chunks: int
    used_ocr: bool = False


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3)
    document_id: Optional[str] = None
    top_k: int = Field(default=5, ge=1, le=10)
    session_id: Optional[str] = None
    model_profile: str = Field(default="balanced")
    generation_model: Optional[str] = None


class Citation(BaseModel):
    document_id: str
    filename: str
    page: int
    chunk_id: int
    score: float
    excerpt: str


class StructuredAnswer(BaseModel):
    direct_answer: str
    key_points: List[str]
    limitations: List[str]


class AskResponse(BaseModel):
    answer: str
    structured_answer: StructuredAnswer
    confidence: float
    citations: List[Citation]
    session_id: str
    agent_trace: List[str]
    model_used: str
    model_profile: str


class EvaluateRequest(BaseModel):
    question: str = Field(..., min_length=3)
    reference_answer: str = Field(..., min_length=1)
    document_id: Optional[str] = None
    top_k: int = Field(default=5, ge=1, le=10)


class EvaluateResponse(BaseModel):
    generated_answer: str
    lexical_f1: float
    semantic_similarity: float
    grounded_confidence: float


class HistoryTurn(BaseModel):
    role: str
    content: str


class SessionHistoryResponse(BaseModel):
    session_id: str
    turns: List[HistoryTurn]
