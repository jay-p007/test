from dataclasses import dataclass
from typing import Dict, List

from app.models import HistoryTurn


@dataclass
class DocumentMeta:
    document_id: str
    filename: str
    pages: int
    chunks: int


class InMemoryDocumentStore:
    def __init__(self) -> None:
        self._documents: Dict[str, DocumentMeta] = {}

    def upsert(self, meta: DocumentMeta) -> None:
        self._documents[meta.document_id] = meta

    def get(self, document_id: str) -> DocumentMeta:
        if document_id not in self._documents:
            raise KeyError(f"Document {document_id} not found")
        return self._documents[document_id]

    def has_any(self) -> bool:
        return bool(self._documents)


class ConversationStore:
    def __init__(self, max_turns: int = 10) -> None:
        self._sessions: Dict[str, List[HistoryTurn]] = {}
        self.max_turns = max_turns

    def append_turn(self, session_id: str, role: str, content: str) -> None:
        turns = self._sessions.setdefault(session_id, [])
        turns.append(HistoryTurn(role=role, content=content))
        if len(turns) > (self.max_turns * 2):
            self._sessions[session_id] = turns[-(self.max_turns * 2) :]

    def get_history(self, session_id: str) -> List[HistoryTurn]:
        return list(self._sessions.get(session_id, []))
