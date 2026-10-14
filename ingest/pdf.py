"""PDF ingestion via pypdf. Encrypted/unreadable PDFs are skipped with a log."""
import sys

from pypdf import PdfReader

from .base import Document


def ingest_pdf(abs_path, rel_path):
    docs = []
    try:
        reader = PdfReader(abs_path)
    except Exception as e:  # corrupt / unreadable
        print(f"[ingest] skip unreadable pdf {rel_path}: {e}", file=sys.stderr)
        return docs
    if reader.is_encrypted:
        try:
            if reader.decrypt("") == 0:
                print(f"[ingest] skip encrypted pdf {rel_path}", file=sys.stderr)
                return docs
        except Exception as e:
            print(f"[ingest] skip encrypted pdf {rel_path}: {e}", file=sys.stderr)
            return docs
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception as e:
            print(f"[ingest] skip page {i} of {rel_path}: {e}", file=sys.stderr)
            continue
        # TODO: scanned PDFs yield no text here -> add OCR fallback per page
        # (render page to image -> images.py OCR path) when sample set shows it.
        if text.strip():
            docs.append(Document(file=rel_path, locator=f"page {i + 1}", text=text))
    return docs
