"""XLSX ingestion via openpyxl. Every sheet becomes a 'Sheet: <name>' section."""
import sys

from openpyxl import load_workbook

from .base import Document


def ingest_xlsx(abs_path, rel_path):
    try:
        wb = load_workbook(abs_path, read_only=True, data_only=True)
    except Exception as e:
        print(f"[ingest] skip unreadable xlsx {rel_path}: {e}", file=sys.stderr)
        return []
    docs = []
    try:
        for sheet in wb.worksheets:
            lines = [f"Sheet: {sheet.title}"]
            for row in sheet.iter_rows(values_only=True):
                cells = [str(c).strip() for c in row if c is not None and str(c).strip()]
                if cells:
                    lines.append(" | ".join(cells))
            text = "\n".join(lines)
            if len(lines) > 1:  # sheet had at least one data row
                docs.append(Document(file=rel_path, locator=f"Sheet: {sheet.title}", text=text))
    finally:
        wb.close()
    return docs
