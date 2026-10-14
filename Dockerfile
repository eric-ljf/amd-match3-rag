# AMD AI League — Match 3 (RAG) auto-graded container.
# Grader contract:
#   python3 /app/app.py --index /app/corpus                      (<=10 min incl. indexing)
#   python3 /app/app.py --corpus /app/corpus --query-id query_01 --query "..."   (<=30 s/query)
#   -> writes /app/output/<query-id>_output.json  {"answer","citations","confidence"}
# Graded on AMD GPU (--device=/dev/kfd) with NO network. All models must be
# baked into the image at BUILD time.

FROM rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Model caches live inside the image so runtime is fully offline.
    HF_HOME=/app/models \
    TRANSFORMERS_CACHE=/app/models \
    SENTENCE_TRANSFORMERS_HOME=/app/models \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    # Skip easyocr (would try to download its own models at runtime -> offline fail).
    # tesseract handles the OCR fallback and is offline-safe.
    DISABLE_EASYOCR=1

# tesseract for the OCR fallback path (images with printed text).
RUN apt-get update \
 && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-eng \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
# Base image already provides torch (ROCm build); pip sees it as satisfied.
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py config.py chunking.py embeddings.py index_store.py retrieve.py answer.py ./
COPY ingest/ ./ingest/

# Bake the embedding model into the image NOW so grading (offline) works.
# BAAI/bge-large-en-v1.5 (~1.34GB): clears the 1 GiB VRAM floor.
# NOTE: copied from local dir (models/bge-large) instead of downloading,
# because the build container cannot reach huggingface.co.
COPY models/bge-large /app/models/bge-large-en-v1.5/
ENV EMBED_MODEL=/app/models/bge-large-en-v1.5
RUN python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('/app/models/bge-large-en-v1.5')"

# Sanity: CLI parses.
RUN python3 /app/app.py --help

# TODO(LATER):
#  - If dense retrieval on CPU proves too slow for the hidden corpus size,
#    evaluate a ROCm faiss-gpu build (faiss-cpu ignores --device=/dev/kfd).
#  - Bake an LLM reader (small instruct model) here when the extractive
#    baseline is replaced (see answer.py TODO).
