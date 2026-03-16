import json
import re
from typing import Generator, List, Tuple

import google.generativeai as genai
import numpy as np

from app.config import settings
from app.models import Citation, HistoryTurn, StructuredAnswer
from app.services.agent import RetrievalAgent
from app.services.chunker import Chunk
from app.services.vector_db import ChromaVectorDB


class QAService:
    def __init__(self, vector_db: ChromaVectorDB) -> None:
        genai.configure(api_key=settings.gemini_api_key)
        self.vector_db = vector_db
        self.agent = RetrievalAgent()

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        vectors: List[List[float]] = []
        for text in texts:
            response = genai.embed_content(
                model=settings.embedding_model,
                content=text,
                task_type="retrieval_document",
            )
            vectors.append(response["embedding"])
        return np.array(vectors, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        response = genai.embed_content(
            model=settings.embedding_model,
            content=query,
            task_type="retrieval_query",
        )
        return np.array(response["embedding"], dtype=np.float32)

    def index_document(self, document_id: str, filename: str, chunks: List[Chunk]) -> None:
        vectors = self.embed_texts([chunk.text for chunk in chunks])
        self.vector_db.add_document(document_id=document_id, filename=filename, chunks=chunks, vectors=vectors)

    def retrieve(self, question: str, top_k: int, document_id: str | None = None) -> Tuple[List[Citation], float, str]:
        query_vec = self.embed_query(question)
        hits = self.vector_db.search(query_vector=query_vec, top_k=top_k, document_id=document_id)

        citations: List[Citation] = []
        context_parts: List[str] = []
        scores: List[float] = []

        for hit in hits:
            scores.append(hit.score)
            citations.append(
                Citation(
                    document_id=hit.document_id,
                    filename=hit.filename,
                    page=hit.page,
                    chunk_id=hit.chunk_id,
                    score=round(hit.score, 4),
                    excerpt=hit.text[:220],
                )
            )
            context_parts.append(
                f"[doc={hit.document_id} page={hit.page} chunk={hit.chunk_id}] {hit.text}"
            )

        confidence = float(np.mean(scores)) if scores else 0.0
        return citations, confidence, "\n\n".join(context_parts)

    def run_agent(
        self,
        question: str,
        top_k: int,
        document_id: str | None,
        history: List[HistoryTurn],
        model_profile: str = "balanced",
        generation_model: str | None = None,
    ) -> tuple[str, float, List[Citation], List[str], str, str]:
        tuned_top_k, trace = self.agent.plan(question, top_k)
        citations, confidence, context = self.retrieve(question, tuned_top_k, document_id)
        trace.append(f"retriever: fetched {len(citations)} chunks")

        if not citations:
            trace.append("writer: no grounded context found")
            return (
                "I could not find enough context in the indexed documents to answer this.",
                0.0,
                [],
                trace,
                context,
                "",
            )

        answer, model_used = self.generate_answer(
            question=question,
            context=context,
            history=history,
            model_profile=model_profile,
            generation_model=generation_model,
        )
        trace.append(f"writer: drafted answer using {model_used}")

        answer, refine_model = self.refine_answer(
            question=question,
            draft=answer,
            context=context,
            model_profile=model_profile,
            generation_model=generation_model,
        )
        trace.append(f"critic: refined answer using {refine_model}")

        return answer, confidence, citations, trace, context, model_used

    def generate_answer(
        self,
        question: str,
        context: str,
        history: List[HistoryTurn],
        model_profile: str = "balanced",
        generation_model: str | None = None,
    ) -> tuple[str, str]:
        history_text = "\n".join([f"{turn.role}: {turn.content}" for turn in history[-6:]])
        prompt = (
            "You are a document QA assistant. Answer ONLY using the provided context. "
            "If the context does not contain the answer, say that clearly. "
            "Keep the answer concise and factual."
        )
        request_text = (
            f"{prompt}\n\nConversation History:\n{history_text or 'None'}"
            f"\n\nQuestion: {question}\n\nContext:\n{context}"
        )
        return self._generate_with_fallback(
            request_text=request_text,
            model_profile=model_profile,
            generation_model=generation_model,
        )

    def stream_answer(
        self,
        question: str,
        context: str,
        history: List[HistoryTurn],
        model_profile: str = "balanced",
        generation_model: str | None = None,
    ) -> tuple[Generator[str, None, None], str]:
        history_text = "\n".join([f"{turn.role}: {turn.content}" for turn in history[-6:]])
        prompt = (
            "You are a document QA assistant. Answer ONLY using the provided context. "
            "If the context does not contain the answer, say that clearly."
        )
        request_text = (
            f"{prompt}\n\nConversation History:\n{history_text or 'None'}"
            f"\n\nQuestion: {question}\n\nContext:\n{context}"
        )
        models = self._resolve_model_candidates(
            model_profile=model_profile,
            generation_model=generation_model,
        )

        for model_name in models:
            try:
                model = genai.GenerativeModel(model_name)
                stream = model.generate_content(request_text, stream=True)

                def _stream():
                    for chunk in stream:
                        text = getattr(chunk, "text", "") or ""
                        if text:
                            yield text

                return _stream(), model_name
            except Exception as exc:
                if self._is_retryable_model_error(exc):
                    continue
                raise

        raise RuntimeError("All generation models failed due to rate limit or availability.")

    def refine_answer(
        self,
        question: str,
        draft: str,
        context: str,
        model_profile: str = "balanced",
        generation_model: str | None = None,
    ) -> tuple[str, str]:
        prompt = (
            "Improve the draft answer for factual grounding. "
            "Do not add information missing from context. "
            "Return only the final answer text."
        )
        request_text = f"{prompt}\n\nQuestion: {question}\n\nContext:\n{context}\n\nDraft:\n{draft}"
        return self._generate_with_fallback(
            request_text=request_text,
            model_profile=model_profile,
            generation_model=generation_model,
        )

    def evaluate_answer(self, generated_answer: str, reference_answer: str) -> tuple[float, float]:
        lexical_f1 = self._lexical_f1(generated_answer, reference_answer)
        semantic_similarity = self._semantic_similarity(generated_answer, reference_answer)
        return lexical_f1, semantic_similarity

    def to_structured_answer(
        self,
        question: str,
        answer: str,
        context: str,
        model_profile: str = "balanced",
        generation_model: str | None = None,
    ) -> StructuredAnswer:
        prompt = (
            "Convert the answer into STRICT JSON with keys: "
            "direct_answer (string), key_points (array of strings), limitations (array of strings). "
            "No markdown, no extra text."
        )
        request_text = (
            f"{prompt}\n\nQuestion: {question}\n\nContext:\n{context}\n\nAnswer:\n{answer}"
        )
        raw_json, _ = self._generate_with_fallback(
            request_text=request_text,
            model_profile=model_profile,
            generation_model=generation_model,
        )

        parsed = self._parse_json_object(raw_json)
        if parsed:
            try:
                return StructuredAnswer.model_validate(parsed)
            except Exception:
                pass

        return StructuredAnswer(
            direct_answer=answer,
            key_points=[answer] if answer else [],
            limitations=["Generated from available retrieved context."],
        )

    def _semantic_similarity(self, text_a: str, text_b: str) -> float:
        vectors = self.embed_texts([text_a, text_b])
        a, b = vectors[0], vectors[1]
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1.0
        sim = float(np.dot(a, b) / denom)
        return max(0.0, min(1.0, sim))

    def _generate_with_fallback(
        self,
        request_text: str,
        model_profile: str,
        generation_model: str | None = None,
    ) -> tuple[str, str]:
        models = self._resolve_model_candidates(
            model_profile=model_profile,
            generation_model=generation_model,
        )
        last_exc: Exception | None = None

        for model_name in models:
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(request_text)
                return (response.text or "").strip(), model_name
            except Exception as exc:
                last_exc = exc
                if self._is_retryable_model_error(exc):
                    continue
                raise

        raise RuntimeError(f"All generation models failed. Last error: {last_exc}")

    @staticmethod
    def _is_retryable_model_error(exc: Exception) -> bool:
        msg = str(exc).lower()
        return "429" in msg or "resource_exhausted" in msg or "rate limit" in msg

    @staticmethod
    def _csv_to_models(csv_text: str) -> List[str]:
        return [m.strip() for m in csv_text.split(",") if m.strip()]

    def _resolve_model_candidates(self, model_profile: str, generation_model: str | None = None) -> List[str]:
        profile = (model_profile or "balanced").lower().strip()
        if profile == "low_cost":
            candidates = self._csv_to_models(settings.low_cost_models)
        elif profile == "high_quality":
            candidates = self._csv_to_models(settings.high_quality_models)
        else:
            candidates = self._csv_to_models(settings.balanced_models)

        if generation_model and generation_model.strip():
            custom = generation_model.strip()
            candidates = [custom] + [m for m in candidates if m != custom]

        if not candidates:
            candidates = ["gemini-2.5-flash"]

        return candidates

    @staticmethod
    def _parse_json_object(text: str) -> dict | None:
        text = (text or "").strip()
        if not text:
            return None
        try:
            return json.loads(text)
        except Exception:
            start = text.find("{")
            end = text.rfind("}")
            if start == -1 or end == -1 or end <= start:
                return None
            try:
                return json.loads(text[start : end + 1])
            except Exception:
                return None

    @staticmethod
    def _lexical_f1(pred: str, ref: str) -> float:
        pred_tokens = re.findall(r"\w+", pred.lower())
        ref_tokens = re.findall(r"\w+", ref.lower())

        if not pred_tokens or not ref_tokens:
            return 0.0

        pred_counts = {}
        for token in pred_tokens:
            pred_counts[token] = pred_counts.get(token, 0) + 1

        ref_counts = {}
        for token in ref_tokens:
            ref_counts[token] = ref_counts.get(token, 0) + 1

        overlap = 0
        for token, count in pred_counts.items():
            overlap += min(count, ref_counts.get(token, 0))

        precision = overlap / len(pred_tokens)
        recall = overlap / len(ref_tokens)
        if precision + recall == 0:
            return 0.0

        return (2 * precision * recall) / (precision + recall)
