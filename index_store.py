"""Dense index persistence (numpy brute-force inner product).

Replaces faiss.IndexFlatIP: faiss-cpu has no Python 3.14 wheels, and
IndexFlatIP is just brute-force inner product over L2-normalized
embeddings (= cosine similarity). numpy gives bit-identical results.

Layout under INDEX_DIR:
  embeddings.npy - float32 (n_chunks, dim), L2-normalized
  chunks.jsonl   - one JSON per chunk: {file, locator, text}
  meta.json      - {backend, model, dim, n_chunks, files}
                   files: {rel_path: {"withdrawn": bool, "marker": str|None}}
"""
import json
import os

import numpy as np

import config


class NumpyIndex:
    """Minimal faiss.IndexFlatIP-compatible interface over numpy."""

    def __init__(self, embs):
        self.embs = np.ascontiguousarray(embs, dtype=np.float32)

    @property
    def ntotal(self):
        return self.embs.shape[0]

    def search(self, q, k):
        """Return (scores, ids) with the same shapes as faiss: (1, k)."""
        q = np.ascontiguousarray(q, dtype=np.float32)
        scores = self.embs @ q.T  # (n, 1)
        scores = scores[:, 0]
        k = min(k, len(scores))
        # argpartition for top-k, then sort those k by score desc
        idx = np.argpartition(-scores, k - 1)[:k]
        order = np.argsort(-scores[idx], kind="stable")
        top = idx[order]
        return scores[top].reshape(1, -1), top.reshape(1, -1)

    def save(self, path):
        np.save(path, self.embs)

    @classmethod
    def load(cls, path):
        return cls(np.load(path, mmap_mode="r"))


def build_and_save(chunks, backend, index_dir=None, file_meta=None):
    index_dir = index_dir or config.INDEX_DIR
    os.makedirs(index_dir, exist_ok=True)
    texts = [c["text"] for c in chunks]
    embs = backend.encode(texts)
    # L2-normalize so inner product == cosine (backend may already do this;
    # normalizing again is idempotent).
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embs = (embs / norms).astype(np.float32)
    dim = embs.shape[1]
    index = NumpyIndex(embs)
    index.save(os.path.join(index_dir, "embeddings.npy"))
    with open(os.path.join(index_dir, "chunks.jsonl"), "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    with open(os.path.join(index_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump({
            "backend": backend.name,
            "model": getattr(backend, "model_name", None),
            "dim": dim,
            "n_chunks": len(chunks),
            "files": file_meta or {},
        }, f)
    return len(chunks)


def load(index_dir=None):
    index_dir = index_dir or config.INDEX_DIR
    index = NumpyIndex.load(os.path.join(index_dir, "embeddings.npy"))
    chunks = []
    with open(os.path.join(index_dir, "chunks.jsonl"), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    with open(os.path.join(index_dir, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    return index, chunks, meta
