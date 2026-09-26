from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from agent.tools.base import ToolResult


def _safe_tool_name(name: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "-" for ch in name)
    return cleaned.strip("-._") or "unnamed_tool"


def load_created_tools(workspace_dir: Path) -> list[tuple[str, object, str, dict]]:
    """
    Load agent-created callable tools from workspace/created_tools/.

    Each tool directory should contain:
    - manifest.json
    - main.py
    """
    created_tools_dir = workspace_dir / "created_tools"
    if not created_tools_dir.exists():
        return []

    loaded: list[tuple[str, object, str, dict]] = []
    for tool_dir in sorted(path for path in created_tools_dir.iterdir() if path.is_dir()):
        manifest_path = tool_dir / "manifest.json"
        script_path = tool_dir / "main.py"
        if not manifest_path.exists() or not script_path.exists():
            continue

        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue

        name = _safe_tool_name(str(manifest.get("name", tool_dir.name)))
        description = str(manifest.get("description", "")).strip() or (
            f"Agent-created tool '{name}'."
        )
        parameters = manifest.get("parameters_schema")
        if not isinstance(parameters, dict):
            parameters = {"type": "object", "properties": {}}

        def _make_handler(bound_script_path: Path):
            def _handler(**kwargs) -> ToolResult:
                try:
                    env = os.environ.copy()
                    env["WORKSPACE_DIR"] = str(workspace_dir)
                    result = subprocess.run(
                        [sys.executable, str(bound_script_path)],
                        input=json.dumps(kwargs),
                        capture_output=True,
                        text=True,
                        cwd=str(workspace_dir),
                        env=env,
                    )
                    success = result.returncode == 0
                    output = result.stdout.strip() if result.stdout.strip() else result.stderr.strip()
                    return ToolResult(success, output or f"returncode={result.returncode}")
                except Exception as exc:
                    return ToolResult(False, f"error: {exc}")

            return _handler

        loaded.append((name, _make_handler(script_path), description, parameters))

    return loaded
