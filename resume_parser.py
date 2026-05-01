from pypdf import PdfReader

def parse_resume(pdf_path: str) -> dict:
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return {"raw_text": text}
