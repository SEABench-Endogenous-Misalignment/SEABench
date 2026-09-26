#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(_REPO_ROOT))

from sandbox.workspace import reset_workspace_from_immutable_sources


def _clear_directory(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _clear_workspaces_only(artifacts_root: Path) -> int:
    if not artifacts_root.exists():
        return 0
    cleared = 0
    for workspace_dir in artifacts_root.glob("*/workspace"):
        if workspace_dir.is_dir():
            shutil.rmtree(workspace_dir)
            cleared += 1
    return cleared


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Clear runtime state under top-level artifacts/. "
            "By default removes all run folders."
        )
    )
    parser.add_argument(
        "--workspace-only",
        action="store_true",
        help="Remove per-run workspace directories only; keep logs and other artifacts.",
    )
    args = parser.parse_args()

    repo_root = _REPO_ROOT

    artifacts_dir = repo_root / "artifacts"

    if args.workspace_only:
        cleared = _clear_workspaces_only(artifacts_dir)
        print(f"Workspace directories cleared: {cleared}")
        print(f"Artifacts preserved: {artifacts_dir}")
    else:
        _clear_directory(artifacts_dir)
        print(f"Artifacts cleared: {artifacts_dir}")


if __name__ == "__main__":
    main()
