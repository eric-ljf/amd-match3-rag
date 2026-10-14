"""Image ingestion via OCR. easyocr first (if enabled), tesseract fallback.

Grading runs offline, so easyocr (which downloads its own models on first use)
is disabled by default via DISABLE_EASYOCR=1 in the Dockerfile; tesseract is
the offline-safe fallback and is installed in the image.
"""
import sys

from PIL import Image

from .base import Document
import config


def _ocr_easyocr(img):
    import easyocr
    reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    results = reader.readtext(img)
    return "\n".join(t for (_, t, _) in results if t and t.strip())


def _ocr_tesseract(img):
    import pytesseract
    # Diagrams/labels (sparse text) OCR far better with PSM 11; dense pages
    # prefer the default PSM 3. Run both, keep the richer output.
    outs = []
    for psm in (11, 3):
        try:
            t = pytesseract.image_to_string(img, config=f"--psm {psm}")
        except Exception:
            t = ""
        outs.append(t)
    outs.sort(key=lambda t: len(t.split()), reverse=True)
    return outs[0]


def ingest_image(abs_path, rel_path):
    try:
        img = Image.open(abs_path)
        img.load()
    except Exception as e:
        print(f"[ingest] skip unreadable image {rel_path}: {e}", file=sys.stderr)
        return []
    text = ""
    if not config.DISABLE_EASYOCR:
        try:
            text = _ocr_easyocr(img)
        except Exception as e:
            print(f"[ingest] easyocr failed for {rel_path}: {e}", file=sys.stderr)
    if not text.strip():
        try:
            text = _ocr_tesseract(img)
        except Exception as e:
            print(f"[ingest] tesseract failed for {rel_path}: {e}", file=sys.stderr)
            return []
    if not text.strip():
        print(f"[ingest] ocr produced no text for {rel_path}", file=sys.stderr)
        return []
    return [Document(file=rel_path, locator="ocr", text=text)]
