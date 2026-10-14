"""Plain-text-ish ingestion: csv/tsv, txt/log/md/py/json/xml, and unknown extensions.

Unknown extensions are attempted as UTF-8 text; binary-looking content
(null bytes) or decode failures are skipped with a log line.
"""
import csv
import sys

from .base import Document

TEXT_EXTS = {".txt", ".log", ".md", ".markdown", ".py", ".json", ".xml", ".html", ".htm"}


def _looks_binary(data: bytes) -> bool:
    return b"\x00" in data


def ingest_texty(abs_path, rel_path):
    try:
        with open(abs_path, "rb") as f:
            data = f.read()
    except Exception as e:
        print(f"[ingest] skip unreadable {rel_path}: {e}", file=sys.stderr)
        return []
    if not data.strip():
        return []  # empty file -> skip silently
    if _looks_binary(data):
        print(f"[ingest] skip binary {rel_path}", file=sys.stderr)
        return []
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        print(f"[ingest] skip non-utf8 {rel_path}", file=sys.stderr)
        return []
    if not text.strip():
        return []

    ext = rel_path.rsplit(".", 1)[-1].lower() if "." in rel_path else ""
    if ext in ("csv", "tsv"):
        try:
            dialect = "excel-tab" if ext == "tsv" else "excel"
            rows = list(csv.reader(text.splitlines(), dialect=dialect))
            lines = [" | ".join(c.strip() for c in r if c.strip()) for r in rows]
            text = "\n".join(l for l in lines if l)
        except Exception as e:
            print(f"[ingest] csv parse failed for {rel_path}, using raw text: {e}",
                  file=sys.stderr)
    return [Document(file=rel_path, locator="file", text=text)]
