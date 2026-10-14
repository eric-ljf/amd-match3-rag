"""Answer extraction: type-aware value extraction over retrieved chunks.

Grading: `answer` must be the VALUE ONLY; `citations` the EXACT file set
(necessity, not relevance — one extra file zeroes the question).

Pipeline in make_answer():
  1. classify the question -> expected value type
  2. extract typed candidates from hits (fused-score order), trap-aware
  3. if direct extraction is weak, follow ticket IDs (e.g. ORR-1847) found in
     hits with a second retrieval hop, then extract from the merged pool
  4. cite only files that contributed the value (plus the linking file for
     multi-hop); abstain (empty answer, no citations) when nothing qualifies
"""
import re

import config

_TICKET_RE = re.compile(r"\b([A-Z]{2,}-\d+)\b")
_PIN_RE = re.compile(r"\b([A-Z]{1,3}\d{1,3})\b")

_PATTERNS = {
    "quarter": re.compile(r"\bQ[1-4]\s*FY\d+\b", re.IGNORECASE),
    "version": re.compile(r"\b\d+\.\d+\.\d+\b"),
    "part_number": re.compile(r"\b[A-Z0-9]{2,}(?:-[A-Z0-9]+){2,}\b"),
    "error_code": re.compile(r"\b[A-Z]\d{3,}\b"),
    "revision": re.compile(r"\bREV-[A-Z0-9]+\b", re.IGNORECASE),
    "hostname": re.compile(
        r"\b(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,}\b",
        re.IGNORECASE),
    "number": re.compile(r"\b\d+(?:\.\d+)?\b"),
}

_STOP = set(
    "what which whose whom when where is are was were the a an of in on for to "
    "with by from at as that this it and or be do does did have has had not no "
    "so than then too very can will just don there their".split()
)


def _keywords(q):
    return [w for w in re.findall(r"[a-z0-9#]+", q.lower())
            if w not in _STOP and len(w) > 2]


def classify(q):
    lq = q.lower()
    if "quarter" in lq:
        return "quarter"
    if "part number" in lq:
        return "part_number"
    if "firmware" in lq:
        return "version"
    if "error code" in lq:
        return "error_code"
    if re.search(r"\bpin\b", lq):
        return "pin"
    if "revision" in lq:
        return "revision"
    if "temperature" in lq:
        return "temperature"
    if "timeout" in lq or "second" in lq:
        return "timeout"
    if "hostname" in lq or "server" in lq:
        return "hostname"
    if "how many" in lq:
        return "number"
    return "generic"


def _proximity_bonus(text, pos, keywords, window=200):
    """Bonus in [0,1] for being near question keywords."""
    low = text.lower()
    best = window
    for kw in keywords:
        i = low.find(kw)
        while i != -1:
            d = abs(i - pos)
            if d < best:
                best = d
            i = low.find(kw, i + 1)
    return 1.0 - best / window


def _same_line_boost(text, pos, entities):
    """Big boost if the candidate shares a line with a question entity
    (ticket IDs, product names). Catches table rows: the right row wins."""
    line_start = text.rfind("\n", 0, pos) + 1
    line_end = text.find("\n", pos)
    line = text[line_start:line_end if line_end != -1 else len(text)].lower()
    for e in entities:
        if e.lower() in line:
            return 1.0
    return 0.0


def _question_entities(q):
    ents = set(_TICKET_RE.findall(q))
    for m in re.findall(r"\b[A-Z]{2,}-\d+\b", q):
        ents.add(m)
    for m in re.findall(r"\bTQ-\d+\b", q, re.IGNORECASE):
        ents.add(m)
    return ents


def _extract_typed(qtype, question, text, keywords, entities, hit_score):
    """Return list of (score, value, span_pos)."""
    out = []
    if qtype in _PATTERNS:
        rx = _PATTERNS[qtype]
        for m in rx.finditer(text):
            val = m.group(0)
            s = hit_score + 0.3 * _proximity_bonus(text, m.start(), keywords)
            s += 0.8 * _same_line_boost(text, m.start(), entities)
            out.append((s, val, m.start()))
        return out
    if qtype == "temperature":
        # number nearest to a temperature keyword ("junction temperature")
        for m in _PATTERNS["number"].finditer(text):
            s = hit_score + 0.5 * _proximity_bonus(
                text, m.start(), ["temperature", "junction"])
            out.append((s, m.group(0), m.start()))
        return out
    if qtype == "timeout":
        # Prefer CONSTANT = <number> assignments over comments mentioning
        # old values ("Raised from 60" must not beat "DEFAULT_... = 180").
        for m in re.finditer(
                r"^[^#\n]*\b([A-Z_]*TIMEOUT[A-Z_]*)\s*=\s*(\d+)",
                text, re.MULTILINE):
            val = m.group(2)
            s = hit_score + 1.0  # constant assignment outranks prose
            out.append((s, val, m.start(2)))
        if not out:
            for m in _PATTERNS["number"].finditer(text):
                s = hit_score + 0.3 * _proximity_bonus(text, m.start(),
                                                       ["timeout", "second"])
                out.append((s, m.group(0), m.start()))
        return out
    if qtype == "pin":
        v = _extract_pin(question, text)
        if v:
            out.append((hit_score + 0.5, v, 0))
        return out
    if qtype == "generic":
        # Unknown shape: try every typed pattern, let proximity scoring decide.
        for name, rx in _PATTERNS.items():
            for m in rx.finditer(text):
                s = hit_score + 0.3 * _proximity_bonus(text, m.start(),
                                                       keywords)
                s += 0.8 * _same_line_boost(text, m.start(), entities)
                out.append((s, m.group(0), m.start()))
        return out
    return out


_PIN_LINE_RX = re.compile(r"^[A-Z]{1,3}\d{1,3}$")
_SIG_LINE_RX = re.compile(r"^[A-Z][A-Z0-9_#]*$")


def _extract_pin(question, text):
    """Pair pin labels with signal names.

    Diagrams list pins in one run (B11..B16) followed by signal names in a
    second run (GND..GND); the i-th pin carries the i-th signal. Work on the
    token sequence (whitespace-agnostic: chunking collapses newlines, but
    word order is preserved). Take the longest pin run and the N signal-like
    tokens immediately after it — prose mentions ("THERM_ALERT# is asserted
    low...") and stray tokens ("TQ-40") cannot pollute the signal list, and
    the old text-wide scan's "nearest pin before the signal" fallback (which
    answered B16 for THERM_ALERT#, 2026-10-12) is only a last resort.
    """
    sig_m = re.search(r"([A-Z][A-Z0-9_]*#)", question)
    if not sig_m:
        return None
    sig = sig_m.group(1)
    toks = text.split()
    is_pin = lambda t: bool(_PIN_LINE_RX.match(t))
    is_sig = lambda t: bool(_SIG_LINE_RX.match(t)) and not is_pin(t)

    # longest maximal run of consecutive pin tokens
    best_run, i = None, 0
    while i < len(toks):
        if is_pin(toks[i]):
            j = i
            while j < len(toks) and is_pin(toks[j]):
                j += 1
            if best_run is None or j - i > best_run[1] - best_run[0]:
                best_run = (i, j)
            i = j
        else:
            i += 1
    if best_run:
        s, e = best_run
        n = e - s
        cand = toks[e:e + n]
        if len(cand) == n and all(is_sig(t) for t in cand) and sig in cand:
            return toks[s + cand.index(sig)]
    # fallback: nearest pin before the signal mention (old behavior)
    idx = text.find(sig)
    if idx != -1:
        best, bestd = None, None
        for m in _PIN_RE.finditer(text):
            d = idx - m.start()
            if d >= 0 and (bestd is None or d < bestd):
                best, bestd = m.group(1), d
        return best
    return None


def _direct_extract(question, qtype, hits, keywords, entities):
    """Try to extract a value from hits. Returns (value, file, score)."""
    cands = []
    for h in hits:
        for s, v, _ in _extract_typed(qtype, question, h["text"],
                                      keywords, entities, h["fused"]):
            cands.append((v, h["file"], s))
    if not cands:
        return None
    cands.sort(key=lambda x: -x[2])
    return cands[0]


def _refuse():
    return {"answer": "", "citations": [], "confidence": 0.0}


def make_answer(query, hits, followup_search=None):
    """hits: list of dicts {text, file, fused}. followup_search: fn(q)->hits."""
    keywords = _keywords(query)
    entities = _question_entities(query)
    qtype = classify(query)

    if not hits or hits[0]["fused"] < config.ABSTAIN_THRESHOLD:
        return _refuse()
    # topicality guard: the best chunk must share substance with the question
    best_terms = set(re.findall(r"[a-z0-9#]+", hits[0]["text"].lower()))
    if len(set(keywords) & best_terms) < 2 and qtype != "pin":
        # pinout OCR text is terse; pin questions skip this guard
        return _refuse()

    contrib_files = []  # files that contributed, in order
    direct = _direct_extract(query, qtype, hits, keywords, entities)
    value, vfile, vscore = (None, None, 0.0)
    if direct:
        value, vfile, vscore = direct
        contrib_files.append(vfile)

    # multi-hop: follow ticket IDs found in hits but absent from the question.
    # A direct answer pulled from a decoy file (e.g. release notes for a
    # "which release fixed it" question) must not suppress the hop: only a
    # direct answer already grounded in a ticket-mentioning chunk is trusted
    # without hopping (2026-10-12: q09 answered 4.3.0 from release notes
    # while the ticket-linked row said 4.3.2).
    if followup_search is not None:
        seen_tickets = set()
        for h in hits[:3]:
            seen_tickets.update(_TICKET_RE.findall(h["text"]))
        new_tickets = sorted(t for t in seen_tickets if t not in entities)
        if new_tickets:
            grounded = value is not None and any(
                t in h["text"] for h in hits if h["file"] == vfile
                for t in new_tickets)
            if not grounded:
                extra = followup_search(" ".join(new_tickets)) or []
                merged = hits + [h for h in extra
                                 if h["file"] not in {x["file"] for x in hits}]
                retry = _direct_extract(query, qtype, merged, keywords,
                                        entities | set(new_tickets))
                if retry:
                    rvalue, rvfile, rvscore = retry
                    rgrounded = any(
                        t in h["text"] for h in merged
                        if h["file"] == rvfile for t in new_tickets)
                    # linking files: hit chunks (not the value file) that
                    # named the ticket — they are what make the answer
                    # reachable, so they are cited; the ungrounded direct
                    # file is dropped.
                    link_files = []
                    for h in hits[:3]:
                        if h["file"] == rvfile:
                            continue
                        if any(t in h["text"] for t in new_tickets):
                            if h["file"] not in link_files:
                                link_files.append(h["file"])
                    if value is None:
                        if rgrounded:
                            contrib_files = link_files
                            if rvfile not in contrib_files:
                                contrib_files.append(rvfile)
                        else:
                            contrib_files = [rvfile]
                        value, vfile, vscore = rvalue, rvfile, rvscore
                    elif (rvalue, rvfile) != (value, vfile) and rgrounded:
                        contrib_files = link_files
                        if rvfile not in contrib_files:
                            contrib_files.append(rvfile)
                        value, vfile, vscore = rvalue, rvfile, rvscore
                    # else: keep the direct answer and its citations

    if value is None:
        return _refuse()
    return {
        "answer": value,
        "citations": contrib_files,
        "confidence": round(float(min(vscore, 1.0)), 4),
    }
