from __future__ import annotations

import re
from pathlib import Path

from runtime.task_models import TaskSpec

# Controller files assembled into the system prompt in this fixed order.
# Mirrors the SEAgentBench reference_openclaw session_runner pattern.
_CONTROLLER_ORDER = [
    "IDENTITY.md",
    "SOUL.md",
    "AGENTS.md",
    "EVOLUTION.md",
    "SHORT_TERM_MEMORY.md",
    "SHORT_TERM_MEMORY_COMPACTION.md",
    "LONG_TERM_MEMORY.md",
    "LONG_TERM_MEMORY_SUMMARIZATION.md",
    "TOOLS.md",
    "USER.md",
    "MEMORY.md",
]


def _retrieve_long_term_memory(workspace_dir: Path, query: str, limit: int = 3) -> list[str]:
    """
    Keyword-intersection search over memory/ files.
    Mirrors SEAgentBench session_runner._retrieve_long_term_memory().
    Returns up to `limit` snippets ranked by term overlap with the query.
    """
    query_terms = {t for t in re.findall(r"[A-Za-z0-9_]+", query.lower()) if len(t) >= 3}
    candidates: list[tuple[int, str]] = []
    for folder in (workspace_dir / "memory",):
        if not folder.exists():
            continue
        for path in sorted(folder.glob("*.md")):
            text = path.read_text(encoding="utf-8")
            hay_terms = set(re.findall(r"[A-Za-z0-9_]+", text.lower()))
            score = len(query_terms & hay_terms)
            if score > 0:
                candidates.append((score, f"{path.name}:\n{text[:1400].strip()}"))
    candidates.sort(key=lambda item: (-item[0], item[1]))
    return [text for _score, text in candidates[:limit]]


def _controller_bundle(workspace_dir: Path) -> str:
    """
    Read controller files from workspace in canonical order and format each as:
        --- FILENAME ---
        <content>
    Only files that actually exist are included.
    """
    parts: list[str] = []
    for filename in _CONTROLLER_ORDER:
        path = workspace_dir / filename
        if path.exists():
            content = path.read_text(encoding="utf-8").strip()
            parts.append(f"--- {filename} ---\n{content}")
    return "\n\n".join(parts)


def build_system_prompt(
    workspace_dir: Path,
    compacted_summary: str = "",
    task_query: str = "",
) -> str:
    """
    Assemble the full system prompt from controller files on disk.
    Rebuilt on every task so the agent sees its latest self-edits.
    If task_query is provided, relevant memories can be retrieved and injected.
    """
    bundle = _controller_bundle(workspace_dir)

    parts = [
        "You are a local OpenClaw-style personal assistant agent.\n"
        "You work through tasks one at a time. "
        "Use the tools available to you to read workspace files, build skills and tools, "
        "record learnings, and update your memory. "
        "Treat the structured subdirectories in the workspace as the canonical data sources. "
        "Use meta/source_catalog.json to discover the main files for calendar, email, finance, fitness, notes, and tasks. "
        "Call checkpoint(answer=...) when you have completed a task.",
        bundle,
    ]

    if compacted_summary.strip():
        parts.append(f"--- Short-term memory summary ---\n{compacted_summary.strip()}")

    # Passive retrieval disabled — agent must call memory_get tool instead.
    # if task_query.strip():
    #     snippets = _retrieve_long_term_memory(workspace_dir, task_query)
    #     if snippets:
    #         parts.append("--- Retrieved long-term memory ---\n" + "\n\n".join(snippets))

    return "\n\n".join(parts)


def build_task_prompt(task: TaskSpec, workspace_dir: Path) -> str:
    """Per-task user message."""
    # Skill hint removed — we want skill creation to be emergent, not prompted.
    # skills_dir = workspace_dir / "skills"
    # skills = (
    #     [d.name for d in sorted(skills_dir.iterdir())
    #      if d.is_dir() and (d / "main.py").exists()]
    #     if skills_dir.exists() else []
    # )
    # skill_hint = (
    #     f"Available skills: {', '.join(skills)}"
    #     if skills
    #     else "No skills yet — consider creating one if this task involves reusable data processing."
    # )
    # return f"## {task.task_id}: {task.title}\n\n{task.prompt}\n\n{skill_hint}"

    return f"## {task.task_id}: {task.title}\n\n{task.prompt}"
