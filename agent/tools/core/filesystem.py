from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from sandbox.workspace import SandboxGuard
from agent.tools.base import ToolResult
from agent.tools.core.self_improve_gate import CONTROLLER_FILES


def make_filesystem_tools(
    guard: SandboxGuard,
    tracker: Any = None,
    allowed_extension_files: tuple[str, ...] | None = None,
) -> list[tuple[str, Any, str, dict]]:
    """
    Returns (name, handler, description, parameters) tuples ready for registry.register().
    All paths are resolved through the SandboxGuard so the agent cannot escape the workspace.
    """

    def read_file(path: str) -> ToolResult:
        try:
            target = guard.resolve(path)
            if not target.exists():
                return ToolResult(False, f"error: file not found: {path}")
            if target.is_dir():
                return ToolResult(False, f"error: '{path}' is a directory — use list_files")
            return ToolResult(True, target.read_text(encoding="utf-8"))
        except PermissionError as exc:
            return ToolResult(False, str(exc))
        except Exception as exc:
            return ToolResult(False, f"error: {type(exc).__name__}: {exc}")

    _allowed_ext_set: set[str] | None = (
        set(allowed_extension_files) if allowed_extension_files is not None else None
    )

    def write_file(path: str, content: str) -> ToolResult:
        try:
            target = guard.resolve(path)
            if target.name in CONTROLLER_FILES:
                return ToolResult(False, f"error: '{target.name}' is a controller file and cannot be written directly — use apply_controller_update if the active evolution surface allows that file")
            rel = Path(path)
            if _allowed_ext_set is not None and rel.parts and rel.parts[0] == "extensions":
                if rel.as_posix() not in _allowed_ext_set:
                    return ToolResult(False, f"error: '{path}' cannot be written")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            if tracker is not None:
                rel = Path(path)
                if rel.parts[0] == "extensions" and len(rel.parts) > 1:
                    extension_name = rel.parts[1]
                    tracker.log(f"{extension_name}_extension_updated", path=path)
            return ToolResult(True, f"written: {path}")
        except PermissionError as exc:
            return ToolResult(False, str(exc))
        except Exception as exc:
            return ToolResult(False, f"error: {type(exc).__name__}: {exc}")

    def list_files(directory: str = ".") -> ToolResult:
        try:
            target = guard.resolve(directory)
            if not target.exists():
                return ToolResult(False, f"error: directory not found: {directory}")
            if not target.is_dir():
                return ToolResult(False, f"error: '{directory}' is a file — use read_file")
            entries = []
            for item in sorted(target.iterdir()):
                entries.append(item.name + "/" if item.is_dir() else item.name)
            return ToolResult(True, json.dumps(entries))
        except PermissionError as exc:
            return ToolResult(False, str(exc))
        except Exception as exc:
            return ToolResult(False, f"error: {type(exc).__name__}: {exc}")

    def find_files(pattern: str, directory: str = ".", recursive: bool = True, filename_glob: str | None = None) -> ToolResult:
        try:
            target = guard.resolve(directory)
            if not target.exists():
                return ToolResult(False, f"error: directory not found: {directory}")
            if not target.is_dir():
                return ToolResult(False, f"error: '{directory}' is a file — use read_file")

            cmd = ["grep", "-rn" if recursive else "-n", "--with-filename"]
            if filename_glob:
                cmd += ["--include", filename_glob]
            cmd += [pattern, str(target)]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            if result.returncode == 2:
                return ToolResult(False, f"grep error: {result.stderr.strip()}")

            if not result.stdout:
                return ToolResult(True, "no matches found")

            prefix = str(target) + "/"
            lines = [
                line[len(prefix):] if line.startswith(prefix) else line
                for line in result.stdout.splitlines()
            ]
            return ToolResult(True, "\n".join(lines))
        except subprocess.TimeoutExpired:
            return ToolResult(False, "error: search timed out")
        except PermissionError as exc:
            return ToolResult(False, str(exc))
        except Exception as exc:
            return ToolResult(False, f"error: {type(exc).__name__}: {exc}")

    return [
        (
            "read_file",
            read_file,
            "Read the full contents of a file in the workspace.",
            {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": (
                            "Path relative to the workspace root, "
                            "e.g. 'finance/banking/transactions_24m.json' or 'MEMORY.md'."
                        ),
                    }
                },
                "required": ["path"],
            },
        ),
        (
            "write_file",
            write_file,
            "Write content to a file in the workspace. Creates the file and any parent directories if needed.",
            {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path relative to the workspace root.",
                    },
                    "content": {
                        "type": "string",
                        "description": "The content to write.",
                    },
                },
                "required": ["path", "content"],
            },
        ),
        (
            "list_files",
            list_files,
            "List the files and subdirectories inside a workspace directory.",
            {
                "type": "object",
                "properties": {
                    "directory": {
                        "type": "string",
                        "description": (
                            "Directory path relative to workspace root. "
                            "Defaults to '.' (workspace root)."
                        ),
                    }
                },
            },
        ),
        (
            "find_files",
            find_files,
            "Search file contents for a pattern using grep. Returns matching lines as 'file:lineno:line'.",
            {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Grep pattern (string or regex) to search for.",
                    },
                    "directory": {
                        "type": "string",
                        "description": (
                            "Directory path relative to workspace root to search in. "
                            "Defaults to '.' (workspace root)."
                        ),
                    },
                    "recursive": {
                        "type": "boolean",
                        "description": "Whether to search subdirectories recursively. Defaults to true.",
                    },
                    "filename_glob": {
                        "type": "string",
                        "description": "Optional glob to restrict which files are searched, e.g. '*.py'.",
                    },
                },
                "required": ["pattern"],
            },
        ),
    ]
