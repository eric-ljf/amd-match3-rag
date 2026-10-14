#!/usr/bin/env python3
"""Match 3 RAG container entrypoint. Exact grader interface:

    python3 /app/app.py --index /app/corpus
    python3 /app/app.py --corpus /app/corpus --query-id query_01 --query "..."

--index builds the index into /app/index (APP_INDEX_DIR override for tests).
--query loads the index, retrieves, and writes
/app/output/<query-id>_output.json (APP_OUTPUT_DIR override for tests) as
{"answer": ..., "citations": [...], "confidence": ...}.
"""
import argparse
import json
import os
import sys
import time

import config
from ingest import ingest_corpus
from chunking import chunk_documents
from embeddings import get_backend
from index_store import build_and_save, load
from retrieve import hybrid_search
from answer import make_answer


def cmd_index(corpus_dir):
    t0 = time.time()
    docs, file_meta = ingest_corpus(corpus_dir)
    chunks = chunk_documents(docs)
    if not chunks:
        print("[index] WARNING: no chunks produced; writing empty index", file=sys.stderr)
    backend = get_backend()
    n = build_and_save(chunks, backend, file_meta=file_meta)
    dt = time.time() - t0
    print(f"[index] done: {n} chunks in {dt:.1f}s -> {config.INDEX_DIR}", file=sys.stderr)


def cmd_query(corpus_dir, query_id, query):
    t0 = time.time()
    backend = get_backend()
    index, chunks, meta = load()
    if meta.get("backend") != backend.name:
        print(f"[query] WARNING: index built with backend={meta.get('backend')}, "
              f"querying with backend={backend.name}", file=sys.stderr)
    withdrawn = None
    if config.WITHDRAWN_EXCLUDE:
        withdrawn = {f: m.get("marker")
                     for f, m in (meta.get("files") or {}).items()
                     if m.get("withdrawn")}
    hits = hybrid_search(query, index, chunks, backend, withdrawn=withdrawn)

    def _followup(q2):
        return hybrid_search(q2, index, chunks, backend, withdrawn=withdrawn)

    result = make_answer(query, hits, followup_search=_followup)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(config.OUTPUT_DIR, f"{query_id}_output.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)
    dt = time.time() - t0
    print(f"[query] {query_id} -> {out_path} ({dt:.2f}s) "
          f"citations={result['citations']}", file=sys.stderr)
    print(json.dumps(result, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser(description="Match 3 RAG: index and query")
    ap.add_argument("--index", metavar="CORPUS_DIR",
                    help="index mode: build index from CORPUS_DIR")
    ap.add_argument("--corpus", metavar="CORPUS_DIR",
                    help="query mode: corpus dir (citations are relative to it)")
    ap.add_argument("--query-id", metavar="ID",
                    help="query mode: e.g. query_01")
    ap.add_argument("--query", metavar="TEXT",
                    help="query mode: the question text")
    args = ap.parse_args()

    if args.index:
        cmd_index(args.index)
    elif args.query is not None:
        if not args.corpus or not args.query_id:
            ap.error("--query requires --corpus and --query-id")
        cmd_query(args.corpus, args.query_id, args.query)
    else:
        ap.error("need --index CORPUS_DIR or --query ... (with --corpus and --query-id)")


if __name__ == "__main__":
    main()
