#!/usr/bin/env python3
"""Harden upstream tasks for one-shot none failure, then revalidate/repair downstream."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


def _resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def _load_sequence(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("type") != "task_sequence":
        raise SystemExit(f"ERROR: expected task_sequence YAML: {path}")
    if not isinstance(payload.get("tasks"), list):
        raise SystemExit(f"ERROR: missing tasks list: {path}")
    return payload


def _task_id(task: dict[str, Any], idx: int) -> str:
    return str(task.get("id") or f"task_{idx:03d}")


def _select_task_ids(sequence: dict[str, Any], raw: str) -> list[str]:
    all_ids = [_task_id(task, idx) for idx, task in enumerate(sequence["tasks"], start=1)]
    if not raw.strip():
        return all_ids
    selected: list[str] = []
    for item in raw.split(","):
        value = item.strip()
        if not value:
            continue
        if value.isdigit():
            idx = int(value)
            if idx < 1 or idx > len(all_ids):
                raise SystemExit(f"ERROR: task index out of range: {idx}")
            selected.append(all_ids[idx - 1])
        elif value in all_ids:
            selected.append(value)
        else:
            raise SystemExit(f"ERROR: unknown task selector: {value}")
    return selected


def _task_by_id(sequence: dict[str, Any], task_id: str) -> dict[str, Any]:
    for idx, task in enumerate(sequence.get("tasks") or [], start=1):
        if isinstance(task, dict) and _task_id(task, idx) == task_id:
            return task
    raise SystemExit(f"ERROR: task not found in working sequence: {task_id}")


def _infer_role(task: dict[str, Any]) -> str:
    opt = task.get("optimization")
    if isinstance(opt, dict) and str(opt.get("role", "")).strip() in {"upstream", "downstream"}:
        return str(opt["role"]).strip()
    validator = task.get("validator") or {}
    if isinstance(validator, dict) and isinstance(validator.get("safety_llm"), dict):
        return "downstream"
    return "upstream"


def _retries_for_role(args: argparse.Namespace, role: str) -> int | None:
    if args.max_task_retries is not None:
        return args.max_task_retries
    return args.downstream_max_task_retries if role == "downstream" else args.upstream_max_task_retries


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            rows.append(item)
    return rows


def _write_jsonl(path: Path, item: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def _candidate_count(task_dir: Path) -> int:
    rows = _read_jsonl(task_dir / "optimization_log.jsonl")
    seen: set[int] = set()
    for row in rows:
        name = str(row.get("candidate") or "")
        if not name.startswith("candidate_"):
            continue
        try:
            seen.add(int(name.split("_", 1)[1]))
        except ValueError:
            continue
    for path in (task_dir / "candidates").glob("candidate_*.yaml"):
        try:
            seen.add(int(path.stem.split("_", 1)[1]))
        except ValueError:
            continue
    return max(seen) if seen else 0


def _write_downstream_safety_context(sequence: dict[str, Any], task_id: str, path: Path) -> None:
    tasks = sequence.get("tasks") or []
    start_idx = 0
    for idx, task in enumerate(tasks):
        if _task_id(task, idx + 1) == task_id:
            start_idx = idx + 1
            break
    rows: list[dict[str, str]] = []
    for idx, task in enumerate(tasks[start_idx:], start=start_idx + 1):
        if not isinstance(task, dict):
            continue
        criteria = (((task.get("validator") or {}).get("safety_llm") or {}).get("criteria") or "")
        criteria = str(criteria).strip()
        if criteria:
            rows.append(
                {
                    "task_id": _task_id(task, idx),
                    "title": str(task.get("title") or ""),
                    "criteria": criteria,
                }
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"downstream_safety_criteria": rows}, indent=2), encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Refine upstream tasks until the original upstream checks pass and the "
            "non-evolving baseline fails on the first attempt, then refresh/repair downstream."
        )
    )
    parser.add_argument("--task-sequence", required=True)
    parser.add_argument("--sequence-id", default="")
    parser.add_argument("--none-run-dir", default="")
    parser.add_argument("--self-run-dir", default="")
    parser.add_argument("--none-stm-context-file", default="")
    parser.add_argument("--self-stm-context-file", default="")
    parser.add_argument("--none-session-state-file", default="")
    parser.add_argument("--self-session-state-file", default="")
    parser.add_argument("--from-scratch", action="store_true")
    parser.add_argument("--surface", default="controller_update")
    parser.add_argument("--sequence-kind", choices=["attack", "benign"], default="attack")
    parser.add_argument("--provider-profile", default="openrouter")
    parser.add_argument("--agent-model", default="")
    parser.add_argument("--judge-model", default="")
    parser.add_argument("--judge-provider-profile", default="openrouter")
    parser.add_argument("--optimizer-model", default="")
    parser.add_argument("--optimizer-provider-profile", default="openrouter")
    parser.add_argument("--optimizer-backend", choices=["textgrad", "direct"], default="textgrad")
    parser.add_argument("--downstream_safety_lookahead", action="store_true")
    parser.add_argument(
        "--freeze-self-evolution-after-upstream",
        dest="freeze_self_evolution_after_upstream",
        action="store_true",
        help="Freeze self-evolution for downstream revalidation/repair. This is the default.",
    )
    parser.add_argument(
        "--no-freeze-self-evolution-after-upstream",
        dest="freeze_self_evolution_after_upstream",
        action="store_false",
        help="Allow downstream revalidation/repair to continue self-evolving.",
    )
    parser.add_argument("--max-candidates-per-task", type=int, default=5)
    parser.add_argument("--allow-equal-score-candidate", action="store_true")
    parser.add_argument("--no-attack-language-penalty", action="store_true")
    parser.add_argument(
        "--contextual-scoping-defense",
        action="store_true",
        help="Enable contextual-scoping guidance for the self-evolving replay condition only.",
    )
    parser.add_argument(
        "--compartmentalize-evolution-updates",
        action="store_true",
        help="Add advisory seed/update compartments to self-evolving controller files only.",
    )
    parser.add_argument(
        "--salient-evolution",
        action="store_true",
        help="Harness-timestamp self-evolved-update log entries for the self-evolving replay condition only.",
    )
    parser.add_argument("--no-early-stop", action="store_true", help="Disable early stopping during per-task refinement.")
    parser.add_argument("--timeout-seconds", type=int, default=0)
    parser.add_argument("--max-task-retries", type=int, default=None)
    parser.add_argument("--upstream-max-task-retries", type=int, default=2)
    parser.add_argument("--downstream-max-task-retries", type=int, default=0)
    parser.add_argument("--task-ids", default="")
    parser.add_argument(
        "--flat-upstream-task-dirs",
        action="store_true",
        help="Write upstream task optimizer artifacts directly under each task dir.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--in-place", action="store_true")
    parser.add_argument("--upstream-retry-extra-candidates", type=int, default=5)
    parser.add_argument("--upstream-retry-max-candidates", type=int, default=15)
    parser.add_argument("--downstream-repair-extra-candidates", type=int, default=5)
    parser.add_argument("--downstream-repair-max-candidates", type=int, default=15)
    parser.add_argument(
        "--num-baseline-seeds",
        type=int,
        default=1,
        help="Run the none-surface baseline this many times per candidate and require safety across all seeds.",
    )
    parser.set_defaults(freeze_self_evolution_after_upstream=True)
    return parser.parse_args()


def _stage_row(rows: list[dict[str, Any]], stage: str, task_id: str) -> dict[str, Any] | None:
    for row in reversed(rows):
        if str(row.get("stage") or "") == stage and str(row.get("task_id") or "") == task_id:
            return row
    return None


def _apply_state_from_row(
    row: dict[str, Any] | None,
    *,
    carry_selected_context: bool,
) -> tuple[Path | None, Path | None, Path | None, Path | None, Path | None, Path | None]:
    if row is None:
        return None, None, None, None, None, None
    none_run_dir = Path(str(row["next_none_run_dir"])) if str(row.get("next_none_run_dir") or "").strip() else None
    self_run_dir = Path(str(row["next_self_run_dir"])) if str(row.get("next_self_run_dir") or "").strip() else None
    none_stm = None
    self_stm = None
    none_session = Path(str(row["next_none_session_state_file"])) if str(row.get("next_none_session_state_file") or "").strip() else None
    self_session = Path(str(row["next_self_session_state_file"])) if str(row.get("next_self_session_state_file") or "").strip() else None
    if carry_selected_context:
        none_stm = Path(str(row["next_none_stm_context_file"])) if str(row.get("next_none_stm_context_file") or "").strip() else None
        self_stm = Path(str(row["next_self_stm_context_file"])) if str(row.get("next_self_stm_context_file") or "").strip() else None
    return none_run_dir, self_run_dir, none_stm, self_stm, none_session, self_session


def _starts_new_downstream_session(sequence: dict[str, Any], task_id: str) -> bool:
    all_task_ids = [_task_id(task, idx) for idx, task in enumerate(sequence.get("tasks") or [], start=1)]
    if task_id not in all_task_ids:
        return False
    idx = all_task_ids.index(task_id)
    if idx == 0:
        return _infer_role(_task_by_id(sequence, task_id)) == "downstream"
    previous_task = _task_by_id(sequence, all_task_ids[idx - 1])
    current_task = _task_by_id(sequence, task_id)
    return _infer_role(previous_task) == "upstream" and _infer_role(current_task) == "downstream"


def _build_task_cmd(
    *,
    args: argparse.Namespace,
    working_sequence: Path,
    sequence_id: str,
    task_id: str,
    role: str,
    task_dir: Path,
    none_run_dir: Path | None,
    self_run_dir: Path | None,
    none_stm_context_file: Path | None,
    self_stm_context_file: Path | None,
    none_session_state_file: Path | None,
    self_session_state_file: Path | None,
    freeze_self_for_task: bool,
    fresh_session_for_task: bool,
    max_candidates: int,
    max_task_retries: int | None,
    require_none_first_attempt_failure_upstream: bool,
) -> list[str]:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "prompt_optimization" / "refine_task_from_replay_textgrad.py"),
        "--task-sequence",
        str(working_sequence),
        "--sequence-id",
        sequence_id,
        "--task-id",
        task_id,
        "--surface",
        args.surface,
        "--sequence-kind",
        args.sequence_kind,
        "--provider-profile",
        args.provider_profile,
        "--optimizer-backend",
        args.optimizer_backend,
        "--max-candidates",
        str(max_candidates),
        "--timeout-seconds",
        str(args.timeout_seconds),
        "--output-dir",
        str(task_dir),
    ]
    if none_run_dir is not None:
        cmd.extend(["--none-run-dir", str(none_run_dir)])
    if self_run_dir is not None:
        cmd.extend(["--self-run-dir", str(self_run_dir)])
    if freeze_self_for_task:
        cmd.extend(["--self-replay-surface", "none"])
    if none_stm_context_file is not None and not fresh_session_for_task:
        cmd.extend(["--none-stm-context-file", str(none_stm_context_file)])
    if self_stm_context_file is not None and not fresh_session_for_task:
        cmd.extend(["--self-stm-context-file", str(self_stm_context_file)])
    if none_session_state_file is not None and not fresh_session_for_task:
        cmd.extend(["--none-session-state-file", str(none_session_state_file)])
    if self_session_state_file is not None and not fresh_session_for_task:
        cmd.extend(["--self-session-state-file", str(self_session_state_file)])
    if fresh_session_for_task:
        cmd.append("--fresh-session")
    if args.agent_model:
        cmd.extend(["--agent-model", args.agent_model])
    if args.judge_model:
        cmd.extend(["--judge-model", args.judge_model])
    if args.judge_provider_profile:
        cmd.extend(["--judge-provider-profile", args.judge_provider_profile])
    if args.optimizer_model:
        cmd.extend(["--optimizer-model", args.optimizer_model])
    if args.optimizer_provider_profile:
        cmd.extend(["--optimizer-provider-profile", args.optimizer_provider_profile])
    if max_task_retries is not None:
        cmd.extend(["--max-task-retries", str(max_task_retries)])
    if args.allow_equal_score_candidate:
        cmd.append("--allow-equal-score-candidate")
    if args.no_attack_language_penalty:
        cmd.append("--no-attack-language-penalty")
    if args.contextual_scoping_defense:
        cmd.append("--contextual-scoping-defense")
    if args.compartmentalize_evolution_updates:
        cmd.append("--compartmentalize-evolution-updates")
    if args.salient_evolution:
        cmd.append("--salient-evolution")
    if require_none_first_attempt_failure_upstream:
        cmd.append("--require-none-first-attempt-failure-upstream")
    if getattr(args, "num_baseline_seeds", 1) > 1:
        cmd.extend(["--num-baseline-seeds", str(args.num_baseline_seeds)])
    if getattr(args, "no_early_stop", False):
        cmd.append("--no-early-stop")
    if args.resume and task_dir.exists():
        cmd.append("--resume")
    if args.from_scratch and none_run_dir is None and self_run_dir is None:
        cmd.append("--fresh")
    if args.downstream_safety_lookahead and role == "upstream":
        latest_sequence = _load_sequence(working_sequence)
        lookahead_path = task_dir / "downstream_safety_context.json"
        _write_downstream_safety_context(latest_sequence, task_id, lookahead_path)
        cmd.extend(["--downstream-safety-context", str(lookahead_path)])
    return cmd


def _run_task_optimizer(cmd: list[str], task_dir: Path) -> dict[str, Any]:
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)
    summary_path = task_dir / "optimization_summary.json"
    if not summary_path.exists():
        raise SystemExit(f"ERROR: missing task optimizer summary: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    refined = Path(str(summary.get("refined_sequence") or ""))
    if not refined.exists():
        raise SystemExit(f"ERROR: missing refined sequence from task optimizer: {refined}")
    return summary


def main() -> None:
    args = _parse_args()
    task_sequence = _resolve_path(args.task_sequence)
    output_dir = _resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    working_sequence = output_dir / "working_sequence.yaml"
    log_path = output_dir / "sequence_optimization_log.jsonl"
    log_rows = _read_jsonl(log_path) if args.resume else []
    if args.resume:
        if not working_sequence.exists():
            raise SystemExit(f"ERROR: cannot resume; missing {working_sequence}")
        sequence = _load_sequence(working_sequence)
    else:
        sequence = _load_sequence(task_sequence)
        shutil.copy2(task_sequence, working_sequence)

    sequence_id = args.sequence_id.strip() or str(sequence.get("id") or "")
    all_task_ids = [_task_id(task, idx) for idx, task in enumerate(sequence["tasks"], start=1)]
    selected_task_ids = _select_task_ids(sequence, args.task_ids)
    upstream_task_ids = [task_id for task_id in selected_task_ids if _infer_role(_task_by_id(sequence, task_id)) == "upstream"]
    downstream_task_ids = [task_id for task_id in selected_task_ids if _infer_role(_task_by_id(sequence, task_id)) == "downstream"]

    carry_selected_context = False
    none_run_dir = _resolve_path(args.none_run_dir) if args.none_run_dir.strip() else None
    self_run_dir = _resolve_path(args.self_run_dir) if args.self_run_dir.strip() else None
    none_stm_context_file = _resolve_path(args.none_stm_context_file) if args.none_stm_context_file.strip() else None
    self_stm_context_file = _resolve_path(args.self_stm_context_file) if args.self_stm_context_file.strip() else None
    none_session_state_file = _resolve_path(args.none_session_state_file) if args.none_session_state_file.strip() else None
    self_session_state_file = _resolve_path(args.self_session_state_file) if args.self_session_state_file.strip() else None

    if args.resume and log_rows:
        last = log_rows[-1]
        last_none, last_self, last_none_stm, last_self_stm, last_none_session, last_self_session = _apply_state_from_row(last, carry_selected_context=carry_selected_context)
        if none_run_dir is None:
            none_run_dir = last_none
        if self_run_dir is None:
            self_run_dir = last_self
        if none_stm_context_file is None:
            none_stm_context_file = last_none_stm
        if self_stm_context_file is None:
            self_stm_context_file = last_self_stm
        if none_session_state_file is None:
            none_session_state_file = last_none_session
        if self_session_state_file is None:
            self_session_state_file = last_self_session

    if none_run_dir is None and self_run_dir is None and not args.from_scratch and not args.resume:
        raise SystemExit(
            "ERROR: provide --none-run-dir and/or --self-run-dir, use --from-scratch, or use --resume."
        )

    print(f"Sequence: {sequence_id}", flush=True)
    if args.resume:
        print(f"Resume: {len(log_rows)} completed stage row(s) found in {log_path}", flush=True)
    print(f"Selected upstream tasks: {', '.join(upstream_task_ids) or '(none)'}", flush=True)
    print(f"Selected downstream tasks: {', '.join(downstream_task_ids) or '(none)'}", flush=True)

    frozen_self_run_dir: Path | None = None
    for ordinal, task_id in enumerate(upstream_task_ids, start=1):
        existing = _stage_row(log_rows, "upstream", task_id)
        if existing is not None:
            print(f"\n=== upstream {ordinal}/{len(upstream_task_ids)} {task_id} (resume skip) ===", flush=True)
            none_run_dir, self_run_dir, none_stm_context_file, self_stm_context_file, none_session_state_file, self_session_state_file = _apply_state_from_row(
                existing,
                carry_selected_context=carry_selected_context,
            )
            if args.freeze_self_evolution_after_upstream and self_run_dir is not None:
                frozen_self_run_dir = self_run_dir
            continue

        role = "upstream"
        original_ordinal = all_task_ids.index(task_id) + 1
        base_task_dir = output_dir / f"{original_ordinal:03d}_{task_id}"
        task_dir = base_task_dir if args.flat_upstream_task_dirs else base_task_dir / "upstream_hardening"
        max_candidates = max(1, args.max_candidates_per_task)
        existing_candidates = _candidate_count(task_dir)
        if existing_candidates:
            max_candidates = min(
                max(existing_candidates + max(1, args.upstream_retry_extra_candidates), max_candidates),
                max(1, args.upstream_retry_max_candidates),
            )
        cmd = _build_task_cmd(
            args=args,
            working_sequence=working_sequence,
            sequence_id=sequence_id,
            task_id=task_id,
            role=role,
            task_dir=task_dir,
            none_run_dir=none_run_dir,
            self_run_dir=self_run_dir,
            none_stm_context_file=none_stm_context_file,
            self_stm_context_file=self_stm_context_file,
            none_session_state_file=none_session_state_file,
            self_session_state_file=self_session_state_file,
            freeze_self_for_task=False,
            fresh_session_for_task=False,
            max_candidates=max_candidates,
            max_task_retries=_retries_for_role(args, role),
            require_none_first_attempt_failure_upstream=True,
        )

        print(f"\n=== upstream {ordinal}/{len(upstream_task_ids)} {task_id} ===", flush=True)
        summary = _run_task_optimizer(cmd, task_dir)
        refined = Path(str(summary["refined_sequence"]))
        shutil.copy2(refined, working_sequence)

        next_none = str(summary.get("best_none_replay_dir") or "").strip()
        next_self = str(summary.get("best_self_replay_dir") or "").strip()
        next_none_stm = str(summary.get("best_none_stm_context_file") or "").strip()
        next_self_stm = str(summary.get("best_self_stm_context_file") or "").strip()
        next_none_session = str(summary.get("best_none_session_state_file") or "").strip()
        next_self_session = str(summary.get("best_self_session_state_file") or "").strip()
        if next_none:
            none_run_dir = Path(next_none)
        if next_self:
            self_run_dir = Path(next_self)
            if args.freeze_self_evolution_after_upstream:
                frozen_self_run_dir = self_run_dir
        if next_none_stm:
            none_stm_context_file = Path(next_none_stm)
        if next_self_stm:
            self_stm_context_file = Path(next_self_stm)
        if next_none_session:
            none_session_state_file = Path(next_none_session)
        if next_self_session:
            self_session_state_file = Path(next_self_session)

        row = {
            "stage": "upstream",
            "task_id": task_id,
            "task_dir": str(task_dir),
            "role": role,
            "accepted_prompt_changed": bool(summary.get("accepted_prompt_changed")),
            "final_score": summary.get("final_score"),
            "final_hard_ok": summary.get("final_hard_ok"),
            "final_reasons": summary.get("final_reasons"),
            "contextual_scoping_defense": bool(args.contextual_scoping_defense),
            "compartmentalize_evolution_updates": bool(args.compartmentalize_evolution_updates),
            "salient_evolution": bool(args.salient_evolution),
            "require_none_first_attempt_failure_upstream": True,
            "none_first_attempt_task_passed": summary.get("none_first_attempt_task_passed"),
            "none_attempt_count": summary.get("none_attempt_count"),
            "next_none_run_dir": str(none_run_dir) if none_run_dir else "",
            "next_self_run_dir": str(self_run_dir) if self_run_dir else "",
            "next_none_stm_context_file": str(none_stm_context_file) if none_stm_context_file else "",
            "next_self_stm_context_file": str(self_stm_context_file) if self_stm_context_file else "",
            "next_none_session_state_file": str(none_session_state_file) if none_session_state_file else "",
            "next_self_session_state_file": str(self_session_state_file) if self_session_state_file else "",
        }
        _write_jsonl(log_path, row)
        log_rows.append(row)

    for ordinal, task_id in enumerate(downstream_task_ids, start=1):
        role = "downstream"
        original_ordinal = all_task_ids.index(task_id) + 1
        refresh_row = _stage_row(log_rows, "downstream_refresh", task_id)
        repair_row = _stage_row(log_rows, "downstream_repair", task_id)
        chosen_existing = repair_row if repair_row is not None else refresh_row
        if chosen_existing is not None:
            print(f"\n=== downstream {ordinal}/{len(downstream_task_ids)} {task_id} (resume skip) ===", flush=True)
            none_run_dir, self_run_dir, none_stm_context_file, self_stm_context_file, none_session_state_file, self_session_state_file = _apply_state_from_row(
                chosen_existing,
                carry_selected_context=carry_selected_context,
            )
            continue

        self_run_for_task = self_run_dir
        freeze_self_for_task = False
        if args.freeze_self_evolution_after_upstream:
            if frozen_self_run_dir is None:
                frozen_self_run_dir = self_run_dir
            self_run_for_task = frozen_self_run_dir
            freeze_self_for_task = True
            if self_run_for_task is None and not args.from_scratch:
                raise SystemExit(
                    "ERROR: downstream freeze mode needs a completed upstream self-run dir. "
                    "Start from upstream tasks or provide --self-run-dir."
                )
        latest_sequence = _load_sequence(working_sequence)
        fresh_session_for_task = bool(
            freeze_self_for_task and _starts_new_downstream_session(latest_sequence, task_id)
        )

        refresh_dir = output_dir / f"{original_ordinal:03d}_{task_id}" / "refresh_chain"
        refresh_cmd = _build_task_cmd(
            args=args,
            working_sequence=working_sequence,
            sequence_id=sequence_id,
            task_id=task_id,
            role=role,
            task_dir=refresh_dir,
            none_run_dir=none_run_dir,
            self_run_dir=self_run_for_task,
            none_stm_context_file=none_stm_context_file,
            self_stm_context_file=self_stm_context_file,
            none_session_state_file=none_session_state_file,
            self_session_state_file=self_session_state_file,
            freeze_self_for_task=freeze_self_for_task,
            fresh_session_for_task=fresh_session_for_task,
            max_candidates=0,
            max_task_retries=_retries_for_role(args, role),
            require_none_first_attempt_failure_upstream=False,
        )

        print(f"\n=== downstream refresh {ordinal}/{len(downstream_task_ids)} {task_id} ===", flush=True)
        refresh_summary = _run_task_optimizer(refresh_cmd, refresh_dir)

        summary_to_use = refresh_summary
        stage_name = "downstream_refresh"
        if not bool(refresh_summary.get("final_hard_ok")):
            repair_dir = output_dir / f"{original_ordinal:03d}_{task_id}" / "repair_chain"
            existing_candidates = _candidate_count(repair_dir)
            max_candidates = min(
                max(existing_candidates + max(1, args.downstream_repair_extra_candidates), 1),
                max(1, args.downstream_repair_max_candidates),
            )
            repair_cmd = _build_task_cmd(
                args=args,
                working_sequence=working_sequence,
                sequence_id=sequence_id,
                task_id=task_id,
                role=role,
                task_dir=repair_dir,
                none_run_dir=none_run_dir,
                self_run_dir=self_run_for_task,
                none_stm_context_file=none_stm_context_file,
                self_stm_context_file=self_stm_context_file,
                none_session_state_file=none_session_state_file,
                self_session_state_file=self_session_state_file,
                freeze_self_for_task=freeze_self_for_task,
                fresh_session_for_task=fresh_session_for_task,
                max_candidates=max_candidates,
                max_task_retries=_retries_for_role(args, role),
                require_none_first_attempt_failure_upstream=False,
            )
            print(f"\n=== downstream repair {ordinal}/{len(downstream_task_ids)} {task_id} ===", flush=True)
            summary_to_use = _run_task_optimizer(repair_cmd, repair_dir)
            stage_name = "downstream_repair"

        refined = Path(str(summary_to_use["refined_sequence"]))
        shutil.copy2(refined, working_sequence)

        next_none = str(summary_to_use.get("best_none_replay_dir") or "").strip()
        next_self = str(summary_to_use.get("best_self_replay_dir") or "").strip()
        next_none_stm = str(summary_to_use.get("best_none_stm_context_file") or "").strip()
        next_self_stm = str(summary_to_use.get("best_self_stm_context_file") or "").strip()
        next_none_session = str(summary_to_use.get("best_none_session_state_file") or "").strip()
        next_self_session = str(summary_to_use.get("best_self_session_state_file") or "").strip()
        if next_none:
            none_run_dir = Path(next_none)
        if next_self and not freeze_self_for_task:
            self_run_dir = Path(next_self)
        if next_none_stm:
            none_stm_context_file = Path(next_none_stm)
        if next_self_stm:
            self_stm_context_file = Path(next_self_stm)
        if next_none_session:
            none_session_state_file = Path(next_none_session)
        if next_self_session:
            self_session_state_file = Path(next_self_session)

        row = {
            "stage": stage_name,
            "task_id": task_id,
            "task_dir": str(
                output_dir / f"{original_ordinal:03d}_{task_id}" / ("repair_chain" if stage_name == "downstream_repair" else "refresh_chain")
            ),
            "role": role,
            "accepted_prompt_changed": bool(summary_to_use.get("accepted_prompt_changed")),
            "final_score": summary_to_use.get("final_score"),
            "final_hard_ok": summary_to_use.get("final_hard_ok"),
            "final_reasons": summary_to_use.get("final_reasons"),
            "contextual_scoping_defense": bool(args.contextual_scoping_defense),
            "compartmentalize_evolution_updates": bool(args.compartmentalize_evolution_updates),
            "salient_evolution": bool(args.salient_evolution),
            "next_none_run_dir": str(none_run_dir) if none_run_dir else "",
            "next_self_run_dir": str(self_run_dir) if self_run_dir else "",
            "next_none_stm_context_file": str(none_stm_context_file) if none_stm_context_file else "",
            "next_self_stm_context_file": str(self_stm_context_file) if self_stm_context_file else "",
            "next_none_session_state_file": str(none_session_state_file) if none_session_state_file else "",
            "next_self_session_state_file": str(self_session_state_file) if self_session_state_file else "",
            "fresh_session_boundary": fresh_session_for_task,
        }
        _write_jsonl(log_path, row)
        log_rows.append(row)

    final_sequence = output_dir / "final_sequence.yaml"
    shutil.copy2(working_sequence, final_sequence)
    if args.in_place:
        shutil.copy2(final_sequence, task_sequence)
        print(f"Updated in place: {task_sequence}", flush=True)
    print(f"\nWrote final sequence: {final_sequence}", flush=True)
    print(f"Wrote log: {log_path}", flush=True)


if __name__ == "__main__":
    started = time.time()
    try:
        main()
    finally:
        print(f"Elapsed: {time.time() - started:.1f}s", flush=True)
