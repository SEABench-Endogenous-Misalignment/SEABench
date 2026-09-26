from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
from typing import Any

import yaml

from runtime.task_models import TaskSetup, TaskSpec


def _task_file_rel_path(path: Path) -> str:
    """Return a stable task-file path rooted at tasks/ when possible."""
    normalized = path.as_posix()
    if normalized.startswith("tasks/"):
        return normalized

    parts = path.parts
    if "tasks" in parts:
        idx = parts.index("tasks")
        return Path(*parts[idx:]).as_posix()

    return normalized


def _task_file_hash(rel_path: str) -> str:
    return hashlib.sha256(rel_path.encode("utf-8")).hexdigest()


def _load_python_validator(ref: str, sequence_path: Path):
    if ":" not in ref:
        raise ValueError(
            f"validator must use '<python_file>:<callable>' format in {sequence_path}: {ref}"
        )

    file_ref, callable_name = ref.split(":", 1)
    file_ref = file_ref.strip()
    callable_name = callable_name.strip()
    if not file_ref or not callable_name:
        raise ValueError(
            f"validator must use '<python_file>:<callable>' format in {sequence_path}: {ref}"
        )

    validator_path = Path(file_ref)
    if not validator_path.is_absolute():
        validator_path = (sequence_path.parent / validator_path).resolve()
    if not validator_path.is_file():
        raise ValueError(f"Validator file not found for {sequence_path}: {validator_path}")

    module_name = (
        f"_seabench_validator_{validator_path.stem}_"
        f"{abs(hash((str(sequence_path), str(validator_path), callable_name)))}"
    )
    spec = importlib.util.spec_from_file_location(module_name, validator_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Unable to load validator module for {sequence_path}: {validator_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    validator = getattr(module, callable_name, None)
    if validator is None:
        raise ValueError(
            f"Validator callable '{callable_name}' not found in {validator_path} for {sequence_path}"
        )
    if not callable(validator):
        raise ValueError(
            f"Validator target '{callable_name}' in {validator_path} is not callable for {sequence_path}"
        )
    return validator


def _infer_task_type_from_rel_path(rel_path: str) -> str:
    """Infer logical task type from tasks/<family>/<type>/... path layout."""
    parts = Path(rel_path).parts
    if len(parts) >= 4 and parts[0] == "tasks":
        return str(parts[2]).strip().lower()
    if len(parts) >= 3 and parts[0] == "tasks":
        if str(parts[2]).lower().endswith(".yaml"):
            return str(parts[1]).strip().lower()
        return str(parts[2]).strip().lower()
    if len(parts) >= 2 and parts[0] == "tasks":
        return str(parts[1]).strip().lower()
    return "generic"


def _parse_setup(raw: Any, label: str, path: Path) -> TaskSetup | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError(f"{label} must be a mapping in {path}")

    setup = TaskSetup()

    controller_overrides = raw.get("controller_overrides")
    if controller_overrides is not None:
        if not isinstance(controller_overrides, dict):
            raise ValueError(f"{label}.controller_overrides must be a mapping in {path}")
        setup.controller_overrides = {str(k): str(v) for k, v in controller_overrides.items()}

    extension_overrides = raw.get("extension_overrides")
    if extension_overrides is not None:
        if not isinstance(extension_overrides, dict):
            raise ValueError(f"{label}.extension_overrides must be a mapping in {path}")
        setup.extension_overrides = {str(k): str(v) for k, v in extension_overrides.items()}

    copy_from_repo = raw.get("copy_from_repo")
    if copy_from_repo is not None:
        if not isinstance(copy_from_repo, list):
            raise ValueError(f"{label}.copy_from_repo must be a list in {path}")
        normalized: list[dict[str, str]] = []
        for item in copy_from_repo:
            if not isinstance(item, dict):
                raise ValueError(f"{label}.copy_from_repo entries must be mappings in {path}")
            normalized.append({"src": str(item.get("src", "")), "dst": str(item.get("dst", ""))})
        setup.copy_from_repo = normalized

    clear_paths = raw.get("clear_paths")
    if clear_paths is not None:
        if not isinstance(clear_paths, list):
            raise ValueError(f"{label}.clear_paths must be a list in {path}")
        setup.clear_paths = [str(p) for p in clear_paths]

    ensure_dirs = raw.get("ensure_dirs")
    if ensure_dirs is not None:
        if not isinstance(ensure_dirs, list):
            raise ValueError(f"{label}.ensure_dirs must be a list in {path}")
        setup.ensure_dirs = [str(p) for p in ensure_dirs]

    tool_allowlist = raw.get("tool_allowlist")
    if tool_allowlist is not None:
        if not isinstance(tool_allowlist, list):
            raise ValueError(f"{label}.tool_allowlist must be a list in {path}")
        setup.tool_allowlist = [str(p) for p in tool_allowlist]

    tool_denylist = raw.get("tool_denylist")
    if tool_denylist is not None:
        if not isinstance(tool_denylist, list):
            raise ValueError(f"{label}.tool_denylist must be a list in {path}")
        setup.tool_denylist = [str(p) for p in tool_denylist]

    apply_to_reflection = raw.get("apply_tool_limits_to_reflection")
    if apply_to_reflection is not None:
        setup.apply_tool_limits_to_reflection = bool(apply_to_reflection)

    is_effectively_empty = (
        not setup.controller_overrides
        and not setup.extension_overrides
        and not setup.copy_from_repo
        and not setup.clear_paths
        and not setup.ensure_dirs
        and setup.tool_allowlist is None
        and setup.tool_denylist is None
        and setup.apply_tool_limits_to_reflection is True
    )
    return None if is_effectively_empty else setup


def _load_sequence_file(path: Path) -> list[TaskSpec]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Task sequence file must contain a mapping: {path}")

    if payload.get("status", "active") != "active":
        return []

    if payload.get("type") != "task_sequence":
        raise ValueError(f"Unsupported task sequence type in {path}: {payload.get('type')}")

    sequence_id = str(payload.get("id", path.stem))
    sequence_title = str(payload.get("title", sequence_id))
    source_task_file = _task_file_rel_path(path)
    source_task_file_hash = _task_file_hash(source_task_file)
    inferred_task_type = _infer_task_type_from_rel_path(source_task_file)
    sequence_task_type = str(payload.get("task_type", inferred_task_type)).strip().lower() or inferred_task_type
    sequence_validation_backend = str(payload.get("validation_backend", "auto"))
    sequence_setup = _parse_setup(payload.get("sequence_setup"), "sequence_setup", path)
    generation = payload.get("generation")
    target_harm = ""
    if isinstance(generation, dict):
        target_harm = str(generation.get("target_harm") or "").strip().lower()
    affordance_files: list[str] = []
    if isinstance(generation, dict) and isinstance(generation.get("affordance_files"), list):
        affordance_files = [
            str(item).strip()
            for item in generation.get("affordance_files", [])
            if str(item).strip()
        ]
    tasks = payload.get("tasks", [])
    if not isinstance(tasks, list):
        raise ValueError(f"'tasks' must be a list in {path}")

    task_specs: list[TaskSpec] = []
    for idx, item in enumerate(tasks, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Task entry #{idx} in {path} must be a mapping")
        task_id = str(item.get("id", f"{sequence_id}-task-{idx:03d}"))
        title = str(item.get("title", task_id))
        prompt = str(item.get("prompt", "")).strip()
        expectation = str(item.get("validator_expectation", "")).strip()
        task_type = str(item.get("task_type", sequence_task_type)).strip().lower() or sequence_task_type
        validation_backend = str(item.get("validation_backend", sequence_validation_backend or "auto"))
        optimization_role = ""
        optimization = item.get("optimization")
        if isinstance(optimization, dict):
            role = str(optimization.get("role") or "").strip().lower()
            if role in {"upstream", "downstream"}:
                optimization_role = role
        if "setup" in item and item.get("setup") is not None:
            raise ValueError(
                f"Per-task setup is not supported. Move it to sequence_setup in {path} (task '{task_id}')."
            )
        validator_raw = item.get("validator", {})
        validator_spec: dict[str, Any] = {}
        python_validator_ref = ""
        python_validator = None
        if validator_raw is None:
            validator_raw = {}
        if isinstance(validator_raw, dict):
            validator_spec = dict(validator_raw)
            safety_llm = validator_spec.get("safety_llm")
            if (
                target_harm == "hallucination"
                and isinstance(safety_llm, dict)
                and str(safety_llm.get("criteria") or "").strip()
            ):
                safety_llm = dict(safety_llm)
                safety_llm["evidence_mode"] = "claim_retrieval"
                validator_spec["safety_llm"] = safety_llm
            if (
                affordance_files
                and isinstance(safety_llm, dict)
                and str(safety_llm.get("criteria") or "").strip()
                and not any(key in safety_llm for key in ("source", "path", "paths"))
            ):
                safety_llm = dict(safety_llm)
                safety_llm["source"] = "files"
                safety_llm["paths"] = list(affordance_files)
                safety_llm["allow_missing_paths"] = True
                validator_spec["safety_llm"] = safety_llm
            python_raw = validator_spec.get("python")
            if python_raw is not None:
                python_validator_ref = str(python_raw).strip()
                if python_validator_ref:
                    python_validator = _load_python_validator(python_validator_ref, path)
                validator_spec.pop("python", None)
        elif isinstance(validator_raw, str):
            python_validator_ref = validator_raw.strip()
            if python_validator_ref:
                python_validator = _load_python_validator(python_validator_ref, path)
        else:
            raise ValueError(f"Task '{task_id}' in {path} has unsupported validator block type")
        if not prompt:
            raise ValueError(f"Task '{task_id}' in {path} is missing a prompt")

        task_specs.append(
            TaskSpec(
                task_id=task_id,
                title=title,
                prompt=prompt,
                task_type=task_type,
                source_task_file=source_task_file,
                source_task_file_hash=source_task_file_hash,
                sequence_id=sequence_id,
                sequence_title=sequence_title,
                optimization_role=optimization_role,
                validator_expectation=expectation,
                validation_backend=validation_backend.strip().lower() or "auto",
                sequence_setup=sequence_setup,
                validator_spec=validator_spec,
                python_validator_ref=python_validator_ref,
                python_validator=python_validator,
            )
        )

    return task_specs


def load_yaml_task_suite(tasks_dir: Path) -> list[TaskSpec]:
    """
    Load all active task-sequence YAML files and flatten them into a task list.

    Files are loaded recursively in path order. Each YAML file represents one sequence.
    """
    task_specs: list[TaskSpec] = []

    for path in sorted(tasks_dir.glob("**/*.yaml")):
        task_specs.extend(_load_sequence_file(path))

    return task_specs


def load_yaml_task_suite_from_files(task_files: list[Path]) -> list[TaskSpec]:
    task_specs: list[TaskSpec] = []
    for path in sorted(task_files):
        task_specs.extend(_load_sequence_file(path))
    return task_specs
