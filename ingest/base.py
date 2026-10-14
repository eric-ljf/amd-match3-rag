"""Core document type shared by all parsers."""
from dataclasses import dataclass


@dataclass
class Document:
    # Citation form: path relative to corpus root, e.g. "docs/report.pdf".
    file: str
    # Where inside the file, e.g. "page 3", "Sheet: Q2", "row 12", "ocr".
    locator: str
    text: str
