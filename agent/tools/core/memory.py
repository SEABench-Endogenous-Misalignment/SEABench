from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from agent.tools.base import ToolResult


def make_memory_tools(
    workspace_dir: Path,
) -> list[tuple[str, Any, str, dict]]:
    """
    Returns (name, handler, description, parameters) tuples for memory retrieval tools.
    Each tool delegates to the agent-editable extension script in the workspace so the
    agent can evolve its own memory retrieval by modifying extensions/memory_*/*.py.
    """

    def _run_extension(extension: str, script: str, query: str, limit: int) -> ToolResult:
        script_path = workspace_dir / "extensions" / extension / script
        if not script_path.exists():
            return ToolResult(False, f"error: extension script not found: {script_path}")
        env = os.environ.copy()
        env["WORKSPACE_DIR"] = str(workspace_dir)
        result = subprocess.run(
            [sys.executable, str(script_path), query, str(limit)],
            capture_output=True,
            text=True,
            env=env,
            cwd=str(workspace_dir),
        )
        if result.returncode != 0:
            return ToolResult(False, f"error: {result.stderr.strip()}")
        return ToolResult(True, result.stdout.strip())

    def memory_get(query: str, limit: int = 3) -> ToolResult:
        return _run_extension("memory_get", "memory_get.py", query, limit)

    def memory_search(query: str, limit: int = 3) -> ToolResult:
        return _run_extension("memory_search", "memory_search.py", query, limit)

    return [
        (
            "memory_get",
            memory_get,
            (
                "Search your long-term memory (memory/) using keyword matching. "
                "Returns up to `limit` file snippets ranked by term overlap with your query. "
                "Call this at the start of a task to retrieve relevant past knowledge."
            ),
            {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Keywords or a short description of what you want to retrieve.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of snippets to return (default 3).",
                    },
                },
                "required": ["query"],
            },
        ),
        (
            "memory_search",
            memory_search,
            (
                "Search your long-term memory (memory/) using semantic similarity. "
                "Returns up to `limit` file snippets ranked by embedding similarity to your query. "
                "Preferred over memory_get when the query is conceptual rather than keyword-based."
            ),
            {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A natural language description of what you want to retrieve.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of snippets to return (default 3).",
                    },
                },
                "required": ["query"],
            },
        ),
    ]
