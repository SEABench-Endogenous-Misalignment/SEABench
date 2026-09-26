# Keyword-based memory retrieval — searches memory/ by term overlap.
# Files are split into overlapping chunks; each chunk is scored separately so that
# relevant passages buried inside long files can still be retrieved.

from __future__ import annotations

import json
import re
from pathlib import Path

_MEMORY_DIRS = ("memory",)
_CHUNK_CHARS = 500
_OVERLAP_CHARS = 100


def _chunk_text(text: str) -> list[str]:
    """Split text into overlapping chunks, breaking on paragraph boundaries."""
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


def memory_get(workspace_dir: Path, query: str, limit: int = 3) -> dict:
    # Returns {"results": ["<filename>:\n<chunk>", ...], "count": N}.
    query_terms = {
        t for t in re.findall(r"[A-Za-z0-9_]+", query.lower()) if len(t) >= 3
    }
    candidates: list[tuple[int, str, str]] = []

    for folder_name in _MEMORY_DIRS:
        folder = workspace_dir / folder_name
        if not folder.exists():
            continue
        for path in sorted(folder.glob("*.md")):
            text = path.read_text(encoding="utf-8")
            for chunk in _chunk_text(text):
                chunk_terms = set(re.findall(r"[A-Za-z0-9_]+", chunk.lower()))
                score = len(query_terms & chunk_terms)
                if score > 0:
                    candidates.append((score, path.name, chunk.strip()))

    candidates.sort(key=lambda item: (-item[0], item[1]))
    snippets = [
        f"{name}:\n{chunk}" for _score, name, chunk in candidates[:limit]
    ]
    return {"results": snippets, "count": len(snippets)}


if __name__ == "__main__":
    import os
    import sys

    ws = Path(os.environ.get("WORKSPACE_DIR", "."))
    q = sys.argv[1] if len(sys.argv) > 1 else ""
    lim = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    print(json.dumps(memory_get(ws, q, lim), indent=2))
