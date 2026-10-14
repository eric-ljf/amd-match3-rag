"""Hybrid retrieval: dense cosine (numpy brute-force IP) + BM25, min-max normalized, weighted fusion.

Withdrawn documents (heuristic flags from ingest/withdrawn.py, stored in index
meta) are excluded from ranking when `withdrawn` is passed as {file: marker}.
This is the citation-safety gate: a withdrawn revision must never be cited,
and one wrongly cited file zeroes the whole question under exact-set scoring.
"""
import re
import sys

from rank_bm25 import BM25Okapi

import config

_WORD_RE = re.compile(r"[a-z0-9]+")


def _tok(s):
    return _WORD_RE.findall(s.lower())


def _minmax(xs):
    lo, hi = min(xs), max(xs)
    if hi - lo < 1e-9:
        return [0.0 for _ in xs]
    return [(x - lo) / (hi - lo) for x in xs]


def hybrid_search(query, index, chunks, backend, top_k=None,
                  w_dense=None, w_bm25=None, withdrawn=None):
    top_k = top_k or config.TOP_K
    w_dense = config.FUSION_W_DENSE if w_dense is None else w_dense
    w_bm25 = config.FUSION_W_BM25 if w_bm25 is None else w_bm25
    n = len(chunks)
    if n == 0:
        return []

    # Withdrawn-document gate: drop their chunk ids from ranking entirely.
    excluded = set()
    if withdrawn:
        for i, c in enumerate(chunks):
            if c["file"] in withdrawn:
                excluded.add(i)
        if excluded:
            for f in sorted({chunks[i]["file"] for i in excluded}):
                print(f"[retrieve] excluded withdrawn document: {f} "
                      f"(marker: {withdrawn.get(f)!r})", file=sys.stderr)
    if len(excluded) >= n:
        return []

    # Dense: cosine via inner product on normalized vectors.
    q = backend.encode([query])
    k = min(max(top_k * 4, top_k), n)
    d_scores, d_ids = index.search(q, k)
    dense = {int(i): float(s) for s, i in zip(d_scores[0], d_ids[0]) if int(i) >= 0}

    # BM25 over all chunks.
    tokenized = [_tok(c["text"]) for c in chunks]
    bm25 = BM25Okapi(tokenized)
    b_scores = bm25.get_scores(_tok(query))
    b_norm = _minmax(list(b_scores))

    cand_ids = set(dense.keys()) | set(range(n))
    cand_ids -= excluded
    d_all = [dense.get(i, 0.0) for i in range(n)]
    d_norm = _minmax(d_all)
    scored = []
    for i in cand_ids:
        fused = w_dense * d_norm[i] + w_bm25 * b_norm[i]
        scored.append((fused, i))
    scored.sort(reverse=True)
    hits = []
    for fused, i in scored[:top_k]:
        hits.append({
            "file": chunks[i]["file"],
            "locator": chunks[i]["locator"],
            "text": chunks[i]["text"],
            "fused": fused,
            "dense": d_norm[i],
            "bm25": b_norm[i],
        })
    return hits
