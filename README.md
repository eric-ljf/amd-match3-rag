# AMD AI League — Match 3 (RAG) container

Auto-graded Docker container: indexes a messy mixed-document corpus, answers
questions with **exact** file citations. One extra/missing cited file = zero
for that question. Unanswerable questions must return an empty answer with
no citations.

## Grader contract

```bash
python3 /app/app.py --index /app/corpus
python3 /app/app.py --corpus /app/corpus --query-id query_01 --query "..."
# -> /app/output/query_01_output.json : {"answer": ..., "citations": [...], "confidence": ...}
```

Limits: ≤10 min startup (incl. indexing), ≤30 s/question, 1–48 GiB VRAM,
≤60 GiB image, **no network during grading** (models baked into the image).
Graded on AMD GPU via `--device=/dev/kfd`; base image is mandated:
`rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0`.

## Layout

```
Dockerfile            # mandated ROCm base; tesseract; pip install; bake MiniLM at build
requirements.txt      # pinned (torch NOT listed: base image already ships ROCm torch)
app.py                # CLI: --index / --query (exact grader interface)
config.py             # env-overridable settings (APP_INDEX_DIR, APP_OUTPUT_DIR, EMB_BACKEND, ...)
ingest/               # per-type parsers -> list[Document{file, locator, text}]
  pdf.py              #   pypdf; encrypted/unreadable skipped w/ stderr log
  docx.py             #   python-docx; tables -> " | "-joined rows
  xlsx.py             #   openpyxl; every sheet -> "Sheet: <name>" section
  csv_txt.py          #   csv/tsv/txt/log/md/py/json/xml + unknown exts as UTF-8 fallback
  images.py           #   PIL + OCR (easyocr if enabled, else tesseract; offline-safe)
chunking.py           # word-based ~512-token chunks, ~64-token overlap (no runtime tokenizer)
embeddings.py         # sbert (real) / hash (deterministic, tests only)
index_store.py        # faiss IndexFlatIP (cosine) + chunks.jsonl + meta.json -> /app/index
retrieve.py           # hybrid: dense cosine + BM25, min-max normalized, weighted fusion
answer.py             # extractive baseline: best chunk -> concise span, cite ONLY its file
tests/                # synthetic corpus + run_test.py (CPU end-to-end)
```

## Pipeline

ingest → chunk → embed (MiniLM-L6-v2, baked at build) → faiss index
→ per query: hybrid retrieve (0.5 dense + 0.5 BM25, top-5) → extractive span
from the single best chunk → cite that chunk's file only → JSON.

Abstention: if the best fused score < `ABSTAIN_THRESHOLD` (default 0.12),
return `{"answer": "", "citations": [], "confidence": 0.0}`.

## Build & run (needs the ROCm base image + GPU host for the real thing)

```bash
docker build -t match3-rag .
# smoke test the grader interface (mount a corpus):
docker run --rm -v /path/to/corpus:/app/corpus:ro --device=/dev/kfd match3-rag \
  python3 /app/app.py --index /app/corpus
docker run --rm -v /path/to/corpus:/app/corpus:ro --device=/dev/kfd match3-rag \
  python3 /app/app.py --corpus /app/corpus --query-id query_01 --query "your question"
```

## Local CPU test (no GPU needed)

```bash
cd ~/workspace/amd-match3-rag
python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU torch first
pip install -r requirements.txt
python3 tests/make_sample_corpus.py
python3 tests/run_test.py            # or: EMB_BACKEND=hash python3 tests/run_test.py
```

The test indexes a synthetic corpus (txt/csv/xlsx/png/empty/unknown-ext/binary),
runs known-answer queries (asserts exact citations), and exercises the
abstention path. OCR of the sample image needs `tesseract` installed locally;
without it the image is skipped (logged) and remaining tests still run.

## Known TODOs (see TODO markers in code)

- `answer.py`: replace extractive baseline with a small baked-in LLM reader;
  multi-file citations only when confident; withdrawn-revision detection.
- `config.py CITATION_FORMAT`: confirm relative-vs-basename against the starter kit —
  the grader compares the cited set exactly.
- `config.py ABSTAIN_THRESHOLD`: calibrate on the official 10-question sample set.
- `ingest/pdf.py`: OCR fallback for scanned PDFs (page -> image -> OCR).
- `Dockerfile`: consider ROCm faiss-gpu if CPU retrieval is too slow on the hidden corpus.
- Starter kit download requires AMD SSO login (researcher could not fetch it);
  grab it at https://lablab.ai/ai-hackathons/amd-lablab-ai-academy-challenge
  for the sample corpus + self-check, then re-run the local test against it.
