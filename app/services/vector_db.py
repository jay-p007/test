from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import chromadb
import numpy as np

from app.config import settings
from app.services.chunker import Chunk


@dataclass
class VectorHit:
    document_id: str
    filename: str
    page: int
    chunk_id: int
    text: str
    score: float


class ChromaVectorDB:
    def __init__(self) -> None:
        self.client = chromadb.PersistentClient(path=settings.chroma_path)
        self.collection = self.client.get_or_create_collection(
            name=settings.chroma_collection,
            metadata={"hnsw:space": "cosine"},
        )

    def add_document(self, document_id: str, filename: str, chunks: List[Chunk], vectors: np.ndarray) -> None:
        ids: List[str] = []
        documents: List[str] = []
        metadatas: List[Dict[str, Any]] = []

        for chunk in chunks:
            ids.append(f"{document_id}:{chunk.chunk_id}")
            documents.append(chunk.text)
            metadatas.append(
                {
                    "document_id": document_id,
                    "filename": filename,
                    "page": int(chunk.page),
                    "chunk_id": int(chunk.chunk_id),
                }
            )

        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=vectors.tolist(),
        )

    def search(self, query_vector: np.ndarray, top_k: int, document_id: Optional[str] = None) -> List[VectorHit]:
        where = {"document_id": document_id} if document_id else None
        result = self.collection.query(
            query_embeddings=[query_vector.tolist()],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        dists = result.get("distances", [[]])[0]

        hits: List[VectorHit] = []
        for text, meta, dist in zip(docs, metas, dists):
            score = max(0.0, 1.0 - float(dist))
            hits.append(
                VectorHit(
                    document_id=str(meta["document_id"]),
                    filename=str(meta["filename"]),
                    page=int(meta["page"]),
                    chunk_id=int(meta["chunk_id"]),
                    text=text,
                    score=score,
                )
            )
        return hits

    def has_any(self) -> bool:
        return self.collection.count() > 0
