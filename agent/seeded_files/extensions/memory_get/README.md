# memory_get

**Type**: Callable tool

Keyword-based search over your long-term memory files (`memory/`). Returns up to `limit` file snippets ranked by term overlap with the query.

## When to use

Use `memory_get` when you know specific keywords you are looking for — task types, tool names, file paths, or domain terms. It is fast and does not require a model to run.

Prefer `memory_search` when your query is conceptual or phrased in natural language.

## Signature

```
memory_get(query: str, limit: int = 3) -> {"results": [...], "count": N}
```

- `query` — keywords or a short description of what you want to retrieve.
- `limit` — maximum number of snippets to return (default 3).

## Returns

JSON object with:
- `results` — list of strings, each formatted as `"<filename>:\n<snippet>"`.
- `count` — number of results returned.
