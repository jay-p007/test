import google.generativeai as genai
import fitz

from app.config import settings


class GeminiOCRService:
    def __init__(self) -> None:
        self.model = genai.GenerativeModel(settings.ocr_model)

    def extract_text_for_page(self, pdf_bytes: bytes, page_number: int) -> str:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        try:
            page = doc.load_page(page_number - 1)
            pix = page.get_pixmap(dpi=180)
            image_bytes = pix.tobytes("png")
        finally:
            doc.close()

        prompt = (
            "Extract all readable text from this document image. "
            "Return plain text only. If unreadable, return an empty string."
        )
        response = self.model.generate_content(
            [
                prompt,
                {"mime_type": "image/png", "data": image_bytes},
            ]
        )
        return " ".join((response.text or "").split())
