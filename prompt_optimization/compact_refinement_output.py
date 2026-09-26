#!/usr/bin/env python3
"""Create a smaller refinement output directory that can still be resumed.

The refinement scripts only need compact evaluation summaries, candidate YAMLs,
sequence logs, checkpoint/workspace state, and exported session/STM state to
resume. Full replay feedback, surface diffs, reflections, answers, and verbose
attempt payloads can make optimization logs enormous, so this script strips those
fields while preserving the keys consumed by the resume path.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any


BIG_METRIC_FIELDS = {"surface_diff", "reflection", "feedback", "safety_feedback"}
METRIC_KEYS = {
    "condition",
    "surface",
    "run_dir",
    "task_passed",
    "first_attempt_task_passed",
    "attempt_count",
    "safety_enabled",
    "safety_passed",
    "surface_update_applied",
    "changed_files",
    "stm_context_file",
    "session_state_file",
}
TASK_RESULT_KEYS = {"passed", "feedback", "surface_update_applied", "surface_diff", "attempts"}
ATTEMPT_KEYS = {
    "passed",
    "surface_update_applied",
    "surface_changed_files",
    "surface_diff",
    "reflection",
}
SAFETY_RESULT_KEYS = {"enabled", "passed", "feedback"}


def _resolve(path: str) -> Path:
    return Path(path).expanduser().resolve()


def _shorten(value: Any, limit: int) -> Any:
    if isinstance(value, str) and limit >= 0 and len(value) > limit:
        return value[:limit] + f"\n...[truncated {len(value) - limit} chars]"
    return value


def _rewrite_paths(value: Any, src: Path, dst: Path) -> Any:
    if isinstance(value, dict):
        return {key: _rewrite_paths(item, src, dst) for key, item in value.items()}
    if isinstance(value, list):
        return [_rewrite_paths(item, src, dst) for item in value]
    if isinstance(value, str):
        src_s = str(src)
        if value == src_s:
            return str(dst)
        if value.startswith(src_s + os.sep):
            return str(dst) + value[len(src_s) :]
    return value


def _compact_metric(value: Any, *, text_limit: int) -> Any:
    if not isinstance(value, dict):
        return value
    out = {key: value.get(key) for key in METRIC_KEYS if key in value}
    for key in BIG_METRIC_FIELDS:
        if key in value:
            out[key] = _shorten(value.get(key, ""), text_limit)
    return out


def _compact_eval(value: Any, *, text_limit: int, src: Path, dst: Path) -> Any:
    value = _rewrite_paths(value, src, dst)
    if not isinstance(value, dict):
        return value
    out = dict(value)
    out["none"] = _compact_metric(out.get("none"), text_limit=text_limit)
    out["self_evolving"] = _compact_metric(out.get("self_evolving"), text_limit=text_limit)
    if isinstance(out.get("lookahead"), dict):
        out["lookahead"] = _rewrite_paths(out["lookahead"], src, dst)
    return out


def _compact_task_result(value: Any, *, text_limit: int, src: Path, dst: Path) -> Any:
    value = _rewrite_paths(value, src, dst)
    if not isinstance(value, dict):
        return value
    out = {key: value.get(key) for key in TASK_RESULT_KEYS if key in value}
    out["feedback"] = _shorten(out.get("feedback", ""), text_limit)
    out["surface_diff"] = _shorten(out.get("surface_diff", ""), text_limit)
    attempts = value.get("attempts")
    if isinstance(attempts, list):
        compact_attempts = []
        for attempt in attempts:
            if not isinstance(attempt, dict):
                continue
            compact = {key: attempt.get(key) for key in ATTEMPT_KEYS if key in attempt}
            compact["surface_diff"] = _shorten(compact.get("surface_diff", ""), text_limit)
            compact["reflection"] = _shorten(compact.get("reflection", ""), text_limit)
            compact_attempts.append(compact)
        out["attempts"] = compact_attempts
    return out


def _compact_safety_result(value: Any, *, text_limit: int, src: Path, dst: Path) -> Any:
    value = _rewrite_paths(value, src, dst)
    if not isinstance(value, dict):
        return value
    out = {key: value.get(key) for key in SAFETY_RESULT_KEYS if key in value}
    out["feedback"] = _shorten(out.get("feedback", ""), text_limit)
    return out


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _compact_json_file(src_file: Path, dst_file: Path, *, src_root: Path, dst_root: Path, text_limit: int) -> None:
    name = src_file.name
    value = _read_json(src_file)
    if name == "final_evaluation.json":
        value = _compact_eval(value, text_limit=text_limit, src=src_root, dst=dst_root)
    elif name == "task_result.json":
        value = _compact_task_result(value, text_limit=text_limit, src=src_root, dst=dst_root)
    elif name == "safety_result.json":
        value = _compact_safety_result(value, text_limit=text_limit, src=src_root, dst=dst_root)
    else:
        value = _rewrite_paths(value, src_root, dst_root)
    _write_json(dst_file, value)


def _compact_jsonl_file(src_file: Path, dst_file: Path, *, src_root: Path, dst_root: Path, text_limit: int) -> None:
    dst_file.parent.mkdir(parents=True, exist_ok=True)
    with src_file.open("r", encoding="utf-8") as reader, dst_file.open("w", encoding="utf-8") as writer:
        for line in reader:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if src_file.name == "optimization_log.jsonl":
                value = _compact_eval(value, text_limit=text_limit, src=src_root, dst=dst_root)
            else:
                value = _rewrite_paths(value, src_root, dst_root)
            writer.write(json.dumps(value, ensure_ascii=False) + "\n")


def _copy_or_link(src_file: Path, dst_file: Path, *, hardlink: bool) -> None:
    dst_file.parent.mkdir(parents=True, exist_ok=True)
    if hardlink:
        os.link(src_file, dst_file)
    else:
        shutil.copy2(src_file, dst_file)


def _should_drop(path: Path, *, drop_event_logs: bool) -> bool:
    return drop_event_logs and "event_logs" in path.parts


def _du_bytes(path: Path) -> int:
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            try:
                total += item.stat().st_size
            except OSError:
                pass
    return total


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compact refinement output while preserving --resume compatibility.")
    parser.add_argument("input_dir", help="Existing refinement output/log directory.")
    parser.add_argument("output_dir", help="Destination for compacted output directory.")
    parser.add_argument("--force", action="store_true", help="Overwrite output_dir if it already exists.")
    parser.add_argument(
        "--copy-files",
        action="store_true",
        help="Physically copy unchanged files instead of hardlinking them. Uses more disk but is portable.",
    )
    parser.add_argument(
        "--text-limit",
        type=int,
        default=2000,
        help="Max chars to retain for bulky text fields. Use 0 for empty strings, -1 to keep full text.",
    )
    parser.add_argument(
        "--drop-event-logs",
        action="store_true",
        help="Omit logs/event_logs files. These are not needed by the refinement resume path.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    src_root = _resolve(args.input_dir)
    dst_root = _resolve(args.output_dir)
    if not src_root.is_dir():
        raise SystemExit(f"ERROR: input_dir is not a directory: {src_root}")
    if dst_root.exists():
        if not args.force:
            raise SystemExit(f"ERROR: output_dir exists; use --force to overwrite: {dst_root}")
        shutil.rmtree(dst_root)
    dst_root.mkdir(parents=True)

    before = _du_bytes(src_root)
    compacted = 0
    linked_or_copied = 0
    dropped = 0
    hardlink = not args.copy_files

    for src_file in src_root.rglob("*"):
        if not src_file.is_file():
            continue
        rel = src_file.relative_to(src_root)
        dst_file = dst_root / rel
        if _should_drop(src_file, drop_event_logs=bool(args.drop_event_logs)):
            dropped += 1
            continue
        try:
            if src_file.suffix == ".jsonl":
                _compact_jsonl_file(src_file, dst_file, src_root=src_root, dst_root=dst_root, text_limit=args.text_limit)
                compacted += 1
            elif src_file.suffix == ".json" and src_file.name in {
                "optimization_summary.json",
                "final_evaluation.json",
                "task_result.json",
                "safety_result.json",
                "replay_meta.json",
            }:
                _compact_json_file(src_file, dst_file, src_root=src_root, dst_root=dst_root, text_limit=args.text_limit)
                compacted += 1
            else:
                _copy_or_link(src_file, dst_file, hardlink=hardlink)
                linked_or_copied += 1
        except Exception as exc:  # noqa: BLE001 - keep utility resilient and explicit.
            raise SystemExit(f"ERROR while processing {src_file}: {exc}") from exc

    after = _du_bytes(dst_root)
    print(f"Input:  {src_root}")
    print(f"Output: {dst_root}")
    print(f"Compacted files: {compacted}")
    print(f"Linked/copied unchanged files: {linked_or_copied}")
    print(f"Dropped files: {dropped}")
    print(f"Apparent input bytes:  {before}")
    print(f"Apparent output bytes: {after}")
    print("Use the compact output dir with --resume in the same way as the original output dir.")


if __name__ == "__main__":
    main()
