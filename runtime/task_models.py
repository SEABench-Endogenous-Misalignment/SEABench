from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


@dataclass
class TaskSetup:
    """
    Optional setup to apply before a sequence starts or before a task runs.

    All paths are workspace-relative unless otherwise noted.
    """

    controller_overrides: dict[str, str] = field(default_factory=dict)
    extension_overrides: dict[str, str] = field(default_factory=dict)
    copy_from_repo: list[dict[str, str]] = field(default_factory=list)  # {src, dst}
    clear_paths: list[str] = field(default_factory=list)
    ensure_dirs: list[str] = field(default_factory=list)
    tool_allowlist: list[str] | None = None
    tool_denylist: list[str] | None = None
    apply_tool_limits_to_reflection: bool = True


@dataclass
class TaskSpec:
    task_id: str
    title: str
    prompt: str
    task_type: str = ""
    source_task_file: str = ""
    source_task_file_hash: str = ""
    sequence_id: str = ""
    sequence_title: str = ""
    optimization_role: str = ""
    validator_expectation: str = ""
    validation_backend: str = "auto"
    sequence_setup: TaskSetup | None = None
    validator_spec: dict[str, Any] = field(default_factory=dict)
    python_validator_ref: str = ""
    python_validator: Callable[[str], tuple[bool, str]] | Callable[[str, Path, Any], tuple[bool, str]] | None = field(
        default=None,
        repr=False,
    )
