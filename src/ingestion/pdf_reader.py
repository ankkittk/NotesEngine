from pypdf import PdfReader
from config import OCR_MIN_TEXT_THRESHOLD


def extract_text_pages(file_path):
    reader = PdfReader(file_path)
    pages_text = []

    for page in reader.pages:
        text = page.extract_text()
        pages_text.append(text if text else "")

    return pages_text


def needs_ocr(text):
    return not text or len(text.strip()) < OCR_MIN_TEXT_THRESHOLD
