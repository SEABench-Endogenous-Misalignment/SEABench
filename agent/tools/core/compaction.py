from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from agent.tools.base import ToolResult


def _run_compaction_extension(
    workspace_dir: Path,
    messages: list[dict[str, Any]],
    previous_summary: str,
    compaction_round: int,
    config: dict[str, Any],
    api_key: str,
    base_url: str | None,
    model: str,
) -> dict[str, Any] | None:
    """
    Run the workspace compaction script via subprocess.
    Returns the parsed stdout dict on success, None on failure.
    """
    script_path = workspace_dir / "extensions" / "compaction" / "compaction.py"
    if not script_path.exists():
        return None

    payload = json.dumps({
        "messages": messages,
        "previous_summary": previous_summary,
        "compaction_round": compaction_round,
        "config": config,
    })

    env = os.environ.copy()
    env["WORKSPACE_DIR"] = str(workspace_dir)
    env["COMPACTION_API_KEY"] = api_key
    env["COMPACTION_MODEL"] = model
    if base_url:
        env["COMPACTION_BASE_URL"] = base_url

    result = subprocess.run(
        [sys.executable, str(script_path)],
        input=payload,
        capture_output=True,
        text=True,
        env=env,
        cwd=str(workspace_dir),
    )

    if result.returncode != 0:
        return None

    try:
        return json.loads(result.stdout.strip())
    except (json.JSONDecodeError, ValueError):
        return None


def make_compaction_tool(
    workspace_dir: Path,
    session: Any,
) -> list[tuple[str, Any, str, dict]]:
    """
    Returns a tool registration tuple for the `compact` tool.
    The agent can call this to explicitly trigger context compaction.
    """

    def compact() -> ToolResult:
        non_system = [m for m in session.messages if m["role"] != "system"]

        result = _run_compaction_extension(
            workspace_dir=workspace_dir,
            messages=non_system,
            previous_summary=session.compacted_summary,
            compaction_round=session.compaction_rounds + 1,
            config=session._compaction_config_dict(),
            api_key=session.config.resolve_api_key(),
            base_url=session.config.base_url,
            model=session.config.model,
        )

        if result is None:
            return ToolResult(False, "error: compaction extension failed or not found")

        if not result.get("dropped"):
            return ToolResult(True, json.dumps({"status": "nothing_to_compact"}))

        session.compacted_summary = result["summary"]
        session.compaction_rounds += 1
        kept = result["kept"]
        dropped = result["dropped"]

        compaction_snapshot_path = session._write_compaction_snapshot()
        system = [m for m in session.messages if m["role"] == "system"]
        session.messages = system[:1] + kept
        session._rebuild_system_prompt()
        session._total_tokens_used = session._estimated_request_tokens()

        session.tracker.log(
            "compaction_round_agent_triggered",
            round=session.compaction_rounds,
            messages_dropped=len(dropped),
            messages_kept=len(kept),
            recent_tokens=sum(session._estimate_message_tokens(m) for m in kept),
            fallback_used=result.get("fallback_used", False),
            memory_path=result.get("memory_path"),
            compaction_snapshot=compaction_snapshot_path,
        )

        return ToolResult(True, json.dumps({
            "status": "compacted",
            "messages_removed": len(dropped),
            "messages_kept": len(kept),
            "memory_path": result.get("memory_path"),
        }))

    return [(
        "compact",
        compact,
        (
            "Explicitly trigger context compaction. Summarises the oldest conversation "
            "messages into the rolling context summary and flushes them to memory, "
            "freeing up space in the active context window. Call this when approaching "
            "context limits or to consolidate completed work before starting a new phase."
        ),
        {
            "type": "object",
            "properties": {},
            "required": [],
        },
    )]
