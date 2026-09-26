# Semantic memory retrieval — searches memory/ by embedding cosine similarity.
# Files are split into overlapping chunks; each chunk is embedded separately so that
# relevant passages buried inside long files can still be retrieved.

from __future__ import annotations

import json
import pickle
import re
from pathlib import Path

import numpy as np

_MEMORY_DIRS = ("memory",)
_CHUNK_CHARS = 500       # target chunk size in characters
_OVERLAP_CHARS = 100     # overlap between consecutive chunks
_MODEL_NAME = "all-MiniLM-L6-v2"


def _chunk_text(text: str) -> list[str]:
    """
    Split text into overlapping chunks, breaking on paragraph boundaries where possible.
    Each chunk is at most _CHUNK_CHARS characters with _OVERLAP_CHARS of overlap.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        if not current:
            current = para
        elif len(current) + len(para) + 2 <= _CHUNK_CHARS:
            current += "\n\n" + para
        else:
            chunks.append(current)
            overlap = current[-_OVERLAP_CHARS:] if len(current) > _OVERLAP_CHARS else current
            current = overlap + "\n\n" + para

    if current:
        chunks.append(current)

    return chunks if chunks else [text[:_CHUNK_CHARS]]


def memory_search(workspace_dir: Path, query: str, limit: int = 3) -> dict:
    # Returns {"results": ["<filename>:\n<chunk>", ...], "count": N}, ranked by similarity.
    from sentence_transformers import SentenceTransformer

    cache_file = workspace_dir / ".embedding_cache.pkl"
    model = SentenceTransformer(_MODEL_NAME)

    files: list[Path] = []
    for folder_name in _MEMORY_DIRS:
        folder = workspace_dir / folder_name
        if not folder.exists():
            continue
        files.extend(sorted(folder.glob("*.md")))

    if not files:
        return {"results": [], "count": 0}

    cache: dict = {}
    if cache_file.exists():
        with cache_file.open("rb") as f:
            cache = pickle.load(f)

    # Build (chunk_text, source_path, cache_key) triples
    all_chunks: list[tuple[str, Path, tuple]] = []
    cache_updated = False

    for path in files:
        mtime = path.stat().st_mtime
        text = path.read_text(encoding="utf-8")
        chunks = _chunk_text(text)
        for idx, chunk in enumerate(chunks):
            key = (str(path), mtime, idx)
            if key not in cache:
                cache[key] = model.encode(chunk).tolist()
                cache_updated = True
            all_chunks.append((chunk, path, key))

    if cache_updated:
        live_keys = {
            (str(p), p.stat().st_mtime, idx)
            for p in files
            for idx, _ in enumerate(_chunk_text(p.read_text(encoding="utf-8")))
        }
        cache = {k: v for k, v in cache.items() if k in live_keys}
        with cache_file.open("wb") as f:
            pickle.dump(cache, f)

    if not all_chunks:
        return {"results": [], "count": 0}

    emb_matrix = np.array([cache[key] for _, _, key in all_chunks])
    query_emb = model.encode(query)
    query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-10)
    emb_norms = emb_matrix / (np.linalg.norm(emb_matrix, axis=1, keepdims=True) + 1e-10)
    scores = emb_norms @ query_norm

    ranked = sorted(zip(scores, all_chunks), key=lambda x: -x[0])

    snippets = [
        f"{path.name}:\n{chunk.strip()}"
        for score, (chunk, path, _) in ranked[:limit]
        if score > 0
    ]

    return {"results": snippets, "count": len(snippets)}


if __name__ == "__main__":
    import os
    import sys

    ws = Path(os.environ.get("WORKSPACE_DIR", "."))
    q = sys.argv[1] if len(sys.argv) > 1 else ""
    lim = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    print(json.dumps(memory_search(ws, q, lim), indent=2))
