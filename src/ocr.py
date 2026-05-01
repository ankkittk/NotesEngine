import fitz
import io
from PIL import Image
import pytesseract


def ocr_document(file_path):
    doc = fitz.open(file_path)
    all_text = []

    for i, page in enumerate(doc):
        print(f"PyMuPDF OCR fallback on page {i + 1}")

        pix = page.get_pixmap()
        img_bytes = pix.tobytes("png")

        image = Image.open(io.BytesIO(img_bytes))
        text = pytesseract.image_to_string(image)

        all_text.append(text)

    return "\n".join(all_text)
