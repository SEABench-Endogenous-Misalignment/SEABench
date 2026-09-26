# memory_search

**Type**: Callable tool

Semantic search over your long-term memory files (`memory/`) using sentence-embedding similarity. Returns up to `limit` file snippets ranked by cosine similarity to the query embedding.

## When to use

Use `memory_search` when your query is conceptual or phrased in natural language rather than specific keywords — e.g. "how did I handle authentication before" rather than "auth token jwt".

Prefer `memory_get` for exact keyword lookups; it is faster and has no model dependency.

## Signature

```
memory_search(query: str, limit: int = 3) -> {"results": [...], "count": N}
```

- `query` — a natural language description of what you want to retrieve.
- `limit` — maximum number of snippets to return (default 3).

## Returns

JSON object with:
- `results` — list of strings, each formatted as `"<filename>:\n<snippet>"`, ordered by descending similarity score.
- `count` — number of results returned.

## Notes

Embeddings are cached in `.embedding_cache.pkl` inside the workspace. The cache is invalidated automatically when a file's modification time changes.
