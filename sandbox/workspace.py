from __future__ import annotations

import shutil
from pathlib import Path

_LEGACY_ENV_EXPORTS = {
    Path("alice_calendar.json"),
    Path("alice_emails.json"),
    Path("alice_banking.json"),
    Path("alice_fitness.json"),
    Path("alice_notes.json"),
}


def _ensure_in_root(root: Path, candidate: Path, original_path: str) -> Path:
    """Return candidate if it is within root, otherwise raise PermissionError."""
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise PermissionError(
            f"Path '{original_path}' escapes sandbox root '{root}'"
        ) from exc
    return candidate


def _ensure_disjoint_paths(workspace_dir: Path, immutable_dir: Path, label: str) -> None:
    """Fail fast if workspace and immutable seed directories overlap."""
    ws = workspace_dir.resolve()
    imm = immutable_dir.resolve()
    if ws == imm or ws in imm.parents or imm in ws.parents:
        raise ValueError(
            f"Workspace '{ws}' must be disjoint from immutable {label} dir '{imm}'"
        )


def _copy_seed_tree(seed_dir: Path, workspace_dir: Path) -> None:
    """Copy the full directory tree from seed_dir into workspace_dir deterministically."""
    for src in sorted(seed_dir.rglob("*")):
        rel = src.relative_to(seed_dir)
        if rel in _LEGACY_ENV_EXPORTS:
            continue
        dst = workspace_dir / rel
        if src.is_dir():
            dst.mkdir(parents=True, exist_ok=True)
            continue
        if src.is_file():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)


class SandboxGuard:
    """Enforces that all file access stays within the workspace root."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def resolve(self, path: str) -> Path:
        # Strip leading slashes so paths like "/bob_calendar.json" are treated as relative
        clean = path.lstrip("/")
        resolved = (self.root / clean).resolve()
        return _ensure_in_root(self.root, resolved, path)


def reset_workspace_from_immutable_sources(
    env_assets_dir: Path,
    workspace_dir: Path,
    controller_surface_dir: Path | None = None,
) -> None:
    """
    Reset the live runtime workspace from immutable seed sources.

    Immutable seed sources:
      - env_assets_dir: data seed (env_assets/)
      - controller_surface_dir: controller seed (repo_root/agent/controller_surface/)

    Mutable runtime destination:
      - workspace_dir: live workspace used by the running agent
    """
    env_assets_dir = env_assets_dir.resolve()
    workspace_dir = workspace_dir.resolve()

    if not env_assets_dir.exists() or not env_assets_dir.is_dir():
        raise FileNotFoundError(f"Environment seed dir not found: {env_assets_dir}")
    _ensure_disjoint_paths(workspace_dir, env_assets_dir, "environment seed")

    if controller_surface_dir is not None:
        controller_surface_dir = controller_surface_dir.resolve()
        if not controller_surface_dir.exists() or not controller_surface_dir.is_dir():
            raise FileNotFoundError(
                f"Controller surface seed dir not found: {controller_surface_dir}"
            )
        _ensure_disjoint_paths(workspace_dir, controller_surface_dir, "controller seed")

    if workspace_dir.exists():
        shutil.rmtree(workspace_dir)
    workspace_dir.mkdir(parents=True)

    _copy_seed_tree(env_assets_dir, workspace_dir)
    if controller_surface_dir is not None:
        _copy_seed_tree(controller_surface_dir, workspace_dir)

    (workspace_dir / ".learnings").mkdir()
    (workspace_dir / "skills").mkdir()
    (workspace_dir / "created_tools").mkdir()
    (workspace_dir / "memory").mkdir()

    memory_path = workspace_dir / "MEMORY.md"
    if not memory_path.exists():
        memory_path.write_text("# Agent Memory\n\n(Nothing recorded yet)\n", encoding="utf-8")


def init_workspace(
    env_assets_dir: Path,
    workspace_dir: Path,
    controller_surface_dir: Path | None = None,
) -> None:
    """
    Copy env_assets into a fresh workspace and create agent directories.

    Environment files are copied into workspace_dir preserving subdirectories.
    Controller surface files (SOUL.md, AGENTS.md, etc.) are seeded from
    controller_surface_dir if provided.
    """
    reset_workspace_from_immutable_sources(
        env_assets_dir=env_assets_dir,
        workspace_dir=workspace_dir,
        controller_surface_dir=controller_surface_dir,
    )
