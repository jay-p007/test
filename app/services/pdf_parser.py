from dataclasses import dataclass
from io import BytesIO
from typing import Callable, List, Optional

from pypdf import PdfReader


@dataclass
class PageText:
    page_number: int
    text: str


def extract_pdf_text(
    pdf_bytes: bytes,
    ocr_page_fn: Optional[Callable[[int], str]] = None,
) -> List[PageText]:
    reader = PdfReader(BytesIO(pdf_bytes))
    pages: List[PageText] = []

    for idx, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = " ".join(text.split())

        if not text and ocr_page_fn is not None:
            text = " ".join((ocr_page_fn(idx) or "").split())

        if text:
            pages.append(PageText(page_number=idx, text=text))

    return pages
