from __future__ import annotations

import shutil
from pathlib import Path

from sandbox.workspace import SandboxGuard
from runtime.task_models import TaskSetup


def _ensure_in_repo_root(repo_root: Path, candidate: Path, original_path: str) -> Path:
    repo_root = repo_root.resolve()
    candidate = candidate.resolve()
    try:
        candidate.relative_to(repo_root)
    except ValueError as exc:
        raise PermissionError(
            f"Repo path '{original_path}' escapes repo root '{repo_root}'"
        ) from exc
    return candidate


def apply_task_setup(
    *,
    workspace_dir: Path,
    repo_root: Path,
    setup: TaskSetup,
) -> None:
    """
    Apply a TaskSetup to the workspace directory.

    This mutates the workspace on disk so subsequent system-prompt rebuilds see the changes.
    """
    guard = SandboxGuard(workspace_dir)

    # 1) Clear requested paths (files or directories).
    for rel in setup.clear_paths:
        target = guard.resolve(rel)
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()

    # 2) Ensure requested directories exist.
    for rel in setup.ensure_dirs:
        target = guard.resolve(rel)
        target.mkdir(parents=True, exist_ok=True)

    # 3) Copy files from the repo into the workspace.
    for item in setup.copy_from_repo:
        src_rel = item.get("src", "").strip()
        dst_rel = item.get("dst", "").strip()
        if not src_rel or not dst_rel:
            continue
        src = _ensure_in_repo_root(repo_root, repo_root / src_rel, src_rel)
        if not src.exists() or not src.is_file():
            raise FileNotFoundError(f"Task setup copy_from_repo src not found: {src_rel}")
        dst = guard.resolve(dst_rel)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    # 4) Apply controller/file overrides (full replacement content).
    for rel, content in setup.controller_overrides.items():
        target = guard.resolve(rel)
        target.parent.mkdir(parents=True, exist_ok=True)
        text = content if content.endswith("\n") else content + "\n"
        target.write_text(text, encoding="utf-8")

    # 5) Apply extension overrides (same mechanism, semantically for extensions/).
    for rel, content in setup.extension_overrides.items():
        target = guard.resolve(rel)
        target.parent.mkdir(parents=True, exist_ok=True)
        text = content if content.endswith("\n") else content + "\n"
        target.write_text(text, encoding="utf-8")

