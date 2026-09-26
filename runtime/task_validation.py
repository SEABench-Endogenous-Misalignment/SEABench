from __future__ import annotations

from pathlib import Path
from typing import Any

from runtime.task_models import TaskSpec


def _resolve_workspace_path(workspace_dir: Path, rel_path: str) -> Path:
    candidate = (workspace_dir / rel_path).resolve()
    root = workspace_dir.resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Workspace check path escapes workspace root: {rel_path}") from exc
    return candidate


def _to_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _run_answer_checks(answer: str, answer_rules: dict[str, Any], failures: list[str]) -> int:
    checks = 0
    answer_l = answer.lower()

    contains_all = answer_rules.get("contains_all", [])
    if contains_all:
        checks += 1
        missing = [s for s in contains_all if str(s).lower() not in answer_l]
        if missing:
            failures.append(f"answer missing required substrings: {missing}")

    contains_any = answer_rules.get("contains_any", [])
    if contains_any:
        checks += 1
        if not any(str(s).lower() in answer_l for s in contains_any):
            failures.append(f"answer missing any-of substrings: {contains_any}")

    not_contains = answer_rules.get("not_contains", [])
    if not_contains:
        checks += 1
        forbidden = [s for s in not_contains if str(s).lower() in answer_l]
        if forbidden:
            failures.append(f"answer contains forbidden substrings: {forbidden}")

    min_length = answer_rules.get("min_length")
    if min_length is not None:
        checks += 1
        if len(answer) < int(min_length):
            failures.append(f"answer shorter than min_length={min_length}")

    return checks


def _run_workspace_checks(
    workspace_dir: Path,
    workspace_checks: list[dict[str, Any]],
    failures: list[str],
) -> int:
    checks = 0

    for idx, check in enumerate(workspace_checks, start=1):
        if not isinstance(check, dict):
            failures.append(f"workspace check #{idx} is not a mapping")
            continue

        check_type = str(check.get("type", "")).strip()
        rel_path = str(check.get("path", "")).strip()
        if not check_type or not rel_path:
            failures.append(f"workspace check #{idx} missing type or path")
            continue

        try:
            target = _resolve_workspace_path(workspace_dir, rel_path)
        except ValueError as exc:
            failures.append(str(exc))
            continue

        if check_type == "file_exists":
            checks += 1
            if not target.is_file():
                failures.append(f"expected file does not exist: {rel_path}")
            continue

        if check_type == "file_not_exists":
            checks += 1
            if target.exists():
                failures.append(f"file should not exist but was found: {rel_path}")
            continue

        if check_type == "dir_exists":
            checks += 1
            if not target.is_dir():
                failures.append(f"expected directory does not exist: {rel_path}")
            continue

        if check_type == "dir_not_exists":
            checks += 1
            if target.exists() and target.is_dir():
                failures.append(f"directory should not exist but was found: {rel_path}")
            continue

        if check_type == "file_contains_all":
            checks += 1
            if not target.is_file():
                failures.append(f"file_contains_all target not found: {rel_path}")
                continue
            text = _to_text(target)
            text_l = text.lower()
            required = [str(s) for s in check.get("contains_all", [])]
            missing = [s for s in required if s.lower() not in text_l]
            if missing:
                failures.append(f"file {rel_path} missing required substrings: {missing}")
            continue

        if check_type == "file_contains_any":
            checks += 1
            if not target.is_file():
                failures.append(f"file_contains_any target not found: {rel_path}")
                continue
            text_l = _to_text(target).lower()
            options = [str(s) for s in check.get("contains_any", [])]
            if options and not any(s.lower() in text_l for s in options):
                failures.append(f"file {rel_path} missing any-of substrings: {options}")
            continue

        failures.append(f"unsupported workspace check type: {check_type}")

    return checks


def evaluate_task_spec_validator(
    task: TaskSpec,
    answer: str,
    workspace_dir: Path,
) -> tuple[bool | None, str]:
    """
    Evaluate deterministic YAML validator rules.

    Returns:
      - (None, "") if no validator rules are configured.
      - (True|False, feedback) if rules were evaluated.
    """
    spec = task.validator_spec or {}
    if not spec:
        return None, ""

    failures: list[str] = []
    checks_run = 0

    answer_rules = spec.get("answer", {})
    if answer_rules:
        if not isinstance(answer_rules, dict):
            return False, "validator.answer must be a mapping"
        checks_run += _run_answer_checks(answer, answer_rules, failures)

    workspace_checks = spec.get("workspace", [])
    if workspace_checks:
        if not isinstance(workspace_checks, list):
            return False, "validator.workspace must be a list"
        checks_run += _run_workspace_checks(workspace_dir, workspace_checks, failures)

    if checks_run == 0:
        return False, "validator provided but no executable checks were found"

    if failures:
        return False, "; ".join(failures)

    return True, f"Passed {checks_run} deterministic checks"
