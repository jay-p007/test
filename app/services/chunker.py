from dataclasses import dataclass
from typing import List

from app.services.pdf_parser import PageText


@dataclass
class Chunk:
    chunk_id: int
    page: int
    text: str


def chunk_pages(pages: List[PageText], chunk_size: int, chunk_overlap: int) -> List[Chunk]:
    chunks: List[Chunk] = []
    chunk_id = 0
    step = max(1, chunk_size - chunk_overlap)

    for page in pages:
        text = page.text
        for start in range(0, len(text), step):
            end = start + chunk_size
            chunk_text = text[start:end].strip()
            if not chunk_text:
                continue

            chunks.append(Chunk(chunk_id=chunk_id, page=page.page_number, text=chunk_text))
            chunk_id += 1

            if end >= len(text):
                break

    return chunks
