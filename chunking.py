"""Word-based chunker approximating ~512 tokens with ~64-token overlap.

Assumption: ~1 token ~= 0.75 words  ->  384 words ~= 512 tokens,
48 words overlap ~= 64 tokens. No tokenizer needed at runtime (offline-safe).
"""
import config


def chunk_text(text, chunk_words=None, overlap_words=None):
    chunk_words = chunk_words or config.CHUNK_WORDS
    overlap_words = overlap_words or config.CHUNK_OVERLAP_WORDS
    words = text.split()
    if not words:
        return []
    chunks = []
    i = 0
    while i < len(words):
        piece = words[i:i + chunk_words]
        chunks.append(" ".join(piece))
        if i + chunk_words >= len(words):
            break
        i += chunk_words - overlap_words
    return chunks


def chunk_documents(docs):
    """docs: list[Document] -> list of dicts {file, locator, text}."""
    out = []
    for doc in docs:
        pieces = chunk_text(doc.text)
        for n, piece in enumerate(pieces):
            locator = doc.locator
            if len(pieces) > 1:
                locator = f"{locator} [part {n + 1}/{len(pieces)}]"
            out.append({"file": doc.file, "locator": locator, "text": piece})
    return out
