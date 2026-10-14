"""Heuristic detection of withdrawn / superseded documents.

Match 3's corpus includes withdrawn revisions as traps: documents that must
NOT be cited even if they textually match a query.

Calibrated against the official sample set (2026-10-08, fixed 2026-10-12):
  * specs/tq40_datasheet_r1_WITHDRAWN.pdf — filename carries WITHDRAWN;
    header reads "Datasheet, revision 1 - WITHDRAWN, SUPERSEDED BY REVISION 2".
  * specs/tq40_datasheet_r2.pdf — header reads "Datasheet, revision 2 -
    supersedes revision 1"; the phrase "Revision 1 is withdrawn" appears
    in the revision-history section, referring to ANOTHER doc. The doc is
    only 534 chars, so a dumb "first 800 chars" header window still caught
    it (2026-10-12 false positive -> q01 abstained).

Lessons encoded below:
  1. Filename markers are strong self-withdrawal signals.
  2. A withdrawal marker is anchored to the CURRENT revision: the doc's own
     revision is parsed from the filename (_r2) or header ("revision 2").
     A marker whose nearest preceding revision reference names a DIFFERENT
     revision ("Revision 1 is withdrawn" in r2's history) is about another
     document and is ignored.
  3. Bare stamps with no revision subject ("WITHDRAWN", "superseded by
     revision 2" in r1's own header) describe the current document.
  4. Active voice "supersedes" (this doc replaces another = current) is NOT
     a withdrawal marker; only passive "superseded (by)" is.
  5. Aggressive keywords ("deprecated", "do not use") are dropped — they
     fire on code comments and ordinary prose.
"""
import re

# Chars of document text scanned for markers (the title/header block).
HEADER_CHARS = 800

# Filename markers: strong signal, the document names its own status.
_FILENAME_PATTERNS = [
    r"withdrawn",
    r"superseded",
    r"obsolete",
    r"revoked",
]

# Header-text markers: passive-voice only. "supersedes" (active) deliberately
# absent — a doc that "supersedes revision 1" is the CURRENT revision.
_HEADER_PATTERNS = [
    r"withdrawn",
    r"superseded",  # bare ("Rev B (superseded)") or "superseded by ..."
    r"obsolete",
    r"revoked",
    r"replaced\s+by",
]

_FILENAME_RX = [re.compile(r"\b" + p + r"\b", re.IGNORECASE)
                for p in _FILENAME_PATTERNS]
_HEADER_RX = [re.compile(r"\b" + p + r"\b", re.IGNORECASE)
              for p in _HEADER_PATTERNS]

# Revision references: "revision 2", "rev 2", "r2" (numeric only; "REV-C2"
# board revisions do not match).
_REV_REF_RX = re.compile(r"\b[rR](?:ev(?:ision)?)?\s*-?\s*(\d+)\b")
# Chars before a marker scanned for its subject revision.
_SUBJECT_WINDOW = 40


def _snippet(text, start, end, ctx=48):
    s = max(0, start - ctx)
    e = min(len(text), end + ctx)
    frag = re.sub(r"\s+", " ", text[s:e]).strip()
    return ("..." if s > 0 else "") + frag + ("..." if e < len(text) else "")


def _own_revision(text, filename):
    """The document's own revision number, if determinable."""
    if filename:
        m = re.search(r"[_-]r(?:ev(?:ision)?)?(\d+)\b", filename,
                      re.IGNORECASE)
        if m:
            return int(m.group(1))
    header = (text or "")[:HEADER_CHARS]
    m = _REV_REF_RX.search(header)
    if m:
        # First revision reference in the header is conventionally the
        # document's own ("Datasheet, revision 2 - ...").
        return int(m.group(1))
    return None


def _marker_subject_revision(header, pos):
    """Nearest revision reference before the marker at pos (its subject)."""
    before = header[max(0, pos - _SUBJECT_WINDOW):pos]
    refs = _REV_REF_RX.findall(before)
    return int(refs[-1]) if refs else None


def detect_withdrawn(text, filename=""):
    """Return (is_withdrawn: bool, marker_snippet: str | None).

    1. Check the filename first (separators normalized to spaces).
    2. Then check the document header (first HEADER_CHARS chars), anchoring
       each marker to the document's own revision: a marker whose subject
       names a different revision is about another document and is skipped.
    First qualifying match wins.
    """
    if filename:
        norm = re.sub(r"[_\-.]+", " ", filename)
        for rx in _FILENAME_RX:
            m = rx.search(norm)
            if m:
                return True, _snippet(norm, m.start(), m.end())
    header = (text or "")[:HEADER_CHARS]
    own = _own_revision(text, filename)
    for rx in _HEADER_RX:
        for m in rx.finditer(header):
            subj = _marker_subject_revision(header, m.start())
            if subj is not None and own is not None and subj != own:
                # "Revision 1 is withdrawn" inside r2's revision history:
                # about another document, not this one.
                continue
            if subj is not None and own is None:
                # Own revision unknown but the marker names a specific
                # revision: revision-history style mention, not a self-stamp.
                continue
            return True, _snippet(header, m.start(), m.end())
    return False, None
