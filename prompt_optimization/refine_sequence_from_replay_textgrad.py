#!/usr/bin/env python3
"""Cascade TextGrad task refinement through a sequence using replayed prior state."""

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


def _write_jsonl(path: Path, item: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(item, ensure_ascii=False) + "\n")


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


def _infer_role(task: dict[str, Any]) -> str:
    opt = task.get("optimization")
    if isinstance(opt, dict) and str(opt.get("role", "")).strip() in {"upstream", "downstream"}:
        return str(opt["role"]).strip()
    validator = task.get("validator") or {}
    if isinstance(validator, dict) and isinstance(validator.get("safety_llm"), dict):
        return "downstream"
    return "upstream"


def _task_by_id(sequence: dict[str, Any], task_id: str) -> dict[str, Any]:
    for idx, task in enumerate(sequence.get("tasks") or [], start=1):
        if isinstance(task, dict) and _task_id(task, idx) == task_id:
            return task
    raise SystemExit(f"ERROR: task not found in working sequence: {task_id}")


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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optimize task_001, replay it, use that state for task_002, and continue."
    )
    parser.add_argument("--task-sequence", required=True, help="Initial task-sequence YAML.")
    parser.add_argument("--sequence-id", default="", help="Sequence id. Defaults to YAML id.")
    parser.add_argument("--none-run-dir", default="", help="Initial prior run dir for surface=none.")
    parser.add_argument("--self-run-dir", default="", help="Initial prior run dir for self-evolving surface.")
    parser.add_argument("--none-stm-context-file", default="", help="Initial selected-path context for surface=none.")
    parser.add_argument("--self-stm-context-file", default="", help="Initial selected-path context for self/evolved surface.")
    parser.add_argument("--none-session-state-file", default="", help="Initial exact session state for surface=none.")
    parser.add_argument("--self-session-state-file", default="", help="Initial exact session state for self/evolved surface.")
    parser.add_argument(
        "--from-scratch",
        action="store_true",
        help="Bootstrap from fresh none and target-surface workspaces instead of initial prior run dirs.",
    )
    parser.add_argument("--surface", default="controller_update", help="Target self-evolution surface.")
    parser.add_argument("--sequence-kind", choices=["attack", "benign"], default="attack")
    parser.add_argument("--provider-profile", default="openrouter")
    parser.add_argument("--agent-model", default="", help="Replay agent model override.")
    parser.add_argument("--judge-model", default="", help="Replay judge model override.")
    parser.add_argument("--optimizer-model", default="", help="Prompt optimizer model override.")
    parser.add_argument("--optimizer-backend", choices=["textgrad", "direct"], default="textgrad")
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
    parser.add_argument(
        "--downstream_safety_lookahead",
        action="store_true",
        help="For upstream tasks, require a cheap judge that the update pertains to later safety criteria.",
    )
    parser.add_argument(
        "--freeze-self-evolution-after-upstream",
        dest="freeze_self_evolution_after_upstream",
        action="store_true",
        help=(
            "For downstream tasks, evaluate the self/evolved condition from the final upstream "
            "self-evolving state while running the downstream replay with surface=none, blocking further evolution. "
            "This is the default."
        ),
    )
    parser.add_argument(
        "--no-freeze-self-evolution-after-upstream",
        dest="freeze_self_evolution_after_upstream",
        action="store_false",
        help="Allow downstream tasks to continue self-evolving instead of freezing after the last upstream task.",
    )
    parser.add_argument("--max-candidates-per-task", type=int, default=1)
    parser.add_argument(
        "--allow-equal-score-candidate",
        action="store_true",
        help="Allow a hard-ok candidate with the same score as the original to replace it.",
    )
    parser.add_argument(
        "--no-attack-language-penalty",
        action="store_true",
        help="Disable the default soft penalty for direct attack/jailbreak-style wording.",
    )
    parser.add_argument("--timeout-seconds", type=int, default=0, help="Per-replay subprocess timeout. Use 0 for no timeout.")
    parser.add_argument("--max-task-retries", type=int, default=None, help="Override retry budget for all tasks.")
    parser.add_argument(
        "--upstream-max-task-retries",
        type=int,
        default=2,
        help="Retry budget for upstream tasks when --max-task-retries is not set. 2 means 1 initial attempt + up to 2 retries.",
    )
    parser.add_argument(
        "--downstream-max-task-retries",
        type=int,
        default=0,
        help="Retry budget for downstream tasks when --max-task-retries is not set. 0 means test each downstream task once.",
    )
    parser.add_argument(
        "--task-ids",
        default="",
        help="Comma-separated task ids or 1-based indices. Defaults to every task in YAML order.",
    )
    parser.add_argument(
        "--only-optimize-downstream",
        action="store_true",
        help=(
            "Optimize only tasks inferred as downstream safety tests. Use with prior --none-run-dir "
            "and --self-run-dir when upstream has already been run."
        ),
    )
    parser.add_argument(
        "--carry-stm-context",
        action="store_true",
        help=(
            "Opt in to carrying selected-path transcript/compaction context between task replays. "
            "By default, refinement resumes from restored workspace and session state without this extra STM handoff."
        ),
    )
    parser.add_argument(
        "--no-carry-selected-context",
        action="store_true",
        help=(
            "Disable the default optimization-only selected-path transcript/compaction context "
            "carry between task replays."
        ),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from output-dir/working_sequence.yaml and sequence_optimization_log.jsonl.",
    )
    parser.add_argument("--in-place", action="store_true", help="Overwrite --task-sequence with the final sequence.")
    parser.set_defaults(freeze_self_evolution_after_upstream=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    task_sequence = _resolve_path(args.task_sequence)
    output_dir = _resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    defense_config = {
        "contextual_scoping_defense": bool(args.contextual_scoping_defense),
        "compartmentalize_evolution_updates": bool(args.compartmentalize_evolution_updates),
        "salient_evolution": bool(args.salient_evolution),
    }
    defense_config_path = output_dir / "defense_config.json"
    if args.resume and defense_config_path.is_file():
        existing = json.loads(defense_config_path.read_text(encoding="utf-8"))
        if existing != defense_config:
            raise SystemExit(
                "ERROR: --resume defence flags differ from the existing output directory: "
                f"existing={existing}, requested={defense_config}"
            )
    defense_config_path.write_text(
        json.dumps(defense_config, indent=2) + "\n",
        encoding="utf-8",
    )

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
    task_ids = _select_task_ids(sequence, args.task_ids)
    if args.only_optimize_downstream:
        task_ids = [
            task_id
            for task_id in task_ids
            if _infer_role(_task_by_id(sequence, task_id)) == "downstream"
        ]
    none_run_dir = _resolve_path(args.none_run_dir) if args.none_run_dir.strip() else None
    self_run_dir = _resolve_path(args.self_run_dir) if args.self_run_dir.strip() else None
    carry_selected_context = bool(args.carry_stm_context) and not args.no_carry_selected_context
    none_stm_context_file = _resolve_path(args.none_stm_context_file) if args.none_stm_context_file.strip() else None
    self_stm_context_file = _resolve_path(args.self_stm_context_file) if args.self_stm_context_file.strip() else None
    none_session_state_file = _resolve_path(args.none_session_state_file) if args.none_session_state_file.strip() else None
    self_session_state_file = _resolve_path(args.self_session_state_file) if args.self_session_state_file.strip() else None
    if args.resume and log_rows:
        last = log_rows[-1]
        if none_run_dir is None and str(last.get("next_none_run_dir") or "").strip():
            none_run_dir = Path(str(last["next_none_run_dir"]))
        if self_run_dir is None and str(last.get("next_self_run_dir") or "").strip():
            self_run_dir = Path(str(last["next_self_run_dir"]))
        if carry_selected_context and none_stm_context_file is None and str(last.get("next_none_stm_context_file") or "").strip():
            none_stm_context_file = Path(str(last["next_none_stm_context_file"]))
        if carry_selected_context and self_stm_context_file is None and str(last.get("next_self_stm_context_file") or "").strip():
            self_stm_context_file = Path(str(last["next_self_stm_context_file"]))
        if none_session_state_file is None and str(last.get("next_none_session_state_file") or "").strip():
            none_session_state_file = Path(str(last["next_none_session_state_file"]))
        if self_session_state_file is None and str(last.get("next_self_session_state_file") or "").strip():
            self_session_state_file = Path(str(last["next_self_session_state_file"]))
        completed = {str(row.get("task_id") or "") for row in log_rows}
        task_ids = [task_id for task_id in task_ids if task_id not in completed]
    if none_run_dir is None and self_run_dir is None and not args.from_scratch:
        raise SystemExit(
            "ERROR: provide --none-run-dir and/or --self-run-dir, use --from-scratch, "
            "or use --resume with a non-empty log."
        )
    if args.from_scratch and none_run_dir is None and self_run_dir is None and task_ids:
        first_idx = all_task_ids.index(task_ids[0]) if task_ids[0] in all_task_ids else -1
        if first_idx != 0:
            raise SystemExit("ERROR: --from-scratch without prior run dirs must start at the first task.")

    frozen_self_run_dir: Path | None = None
    if args.freeze_self_evolution_after_upstream and self_run_dir is not None and task_ids:
        first_task = _task_by_id(sequence, task_ids[0])
        if _infer_role(first_task) == "downstream":
            frozen_self_run_dir = self_run_dir
    print(f"Sequence: {sequence_id}", flush=True)
    if args.resume:
        print(f"Resume: {len(log_rows)} completed task(s) found in {log_path}", flush=True)
    print(f"Tasks: {', '.join(task_ids)}", flush=True)
    if not task_ids:
        print("No remaining tasks selected.", flush=True)
    for ordinal, task_id in enumerate(task_ids, start=1):
        latest_sequence_for_role = _load_sequence(working_sequence)
        role = _infer_role(_task_by_id(latest_sequence_for_role, task_id))
        max_task_retries_for_task = _retries_for_role(args, role)
        self_run_for_task = self_run_dir
        freeze_self_for_task = False
        if args.freeze_self_evolution_after_upstream and role == "downstream":
            if frozen_self_run_dir is None:
                frozen_self_run_dir = self_run_dir
            self_run_for_task = frozen_self_run_dir
            freeze_self_for_task = True
            if self_run_for_task is None and not args.from_scratch:
                raise SystemExit(
                    "ERROR: downstream freeze mode needs a completed upstream self-run dir. "
                    "Start from upstream tasks or provide --self-run-dir."
                )
        fresh_session_for_task = bool(
            freeze_self_for_task and _starts_new_downstream_session(latest_sequence_for_role, task_id)
        )
        original_ordinal = all_task_ids.index(task_id) + 1 if task_id in all_task_ids else ordinal
        task_dir = output_dir / f"{original_ordinal:03d}_{task_id}"
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
            str(max(0, args.max_candidates_per_task)),
            "--timeout-seconds",
            str(args.timeout_seconds),
            "--output-dir",
            str(task_dir),
        ]
        if none_run_dir is not None:
            cmd.extend(["--none-run-dir", str(none_run_dir)])
        if self_run_for_task is not None:
            cmd.extend(["--self-run-dir", str(self_run_for_task)])
        if freeze_self_for_task:
            cmd.extend(["--self-replay-surface", "none"])
        if carry_selected_context and none_stm_context_file is not None:
            if not fresh_session_for_task:
                cmd.extend(["--none-stm-context-file", str(none_stm_context_file)])
        if carry_selected_context and self_stm_context_file is not None:
            if not fresh_session_for_task:
                cmd.extend(["--self-stm-context-file", str(self_stm_context_file)])
        if none_session_state_file is not None:
            if not fresh_session_for_task:
                cmd.extend(["--none-session-state-file", str(none_session_state_file)])
        if self_session_state_file is not None:
            if not fresh_session_for_task:
                cmd.extend(["--self-session-state-file", str(self_session_state_file)])
        if fresh_session_for_task:
            cmd.append("--fresh-session")
        if args.agent_model:
            cmd.extend(["--agent-model", args.agent_model])
        if args.judge_model:
            cmd.extend(["--judge-model", args.judge_model])
        if args.optimizer_model:
            cmd.extend(["--optimizer-model", args.optimizer_model])
        if args.contextual_scoping_defense:
            cmd.append("--contextual-scoping-defense")
        if args.compartmentalize_evolution_updates:
            cmd.append("--compartmentalize-evolution-updates")
        if args.salient_evolution:
            cmd.append("--salient-evolution")
        if max_task_retries_for_task is not None:
            cmd.extend(["--max-task-retries", str(max_task_retries_for_task)])
        if args.allow_equal_score_candidate:
            cmd.append("--allow-equal-score-candidate")
        if args.no_attack_language_penalty:
            cmd.append("--no-attack-language-penalty")
        if args.resume and task_dir.exists():
            cmd.append("--resume")
        fresh_this_task = args.from_scratch and none_run_dir is None and self_run_dir is None
        if fresh_this_task:
            cmd.append("--fresh")
        if args.downstream_safety_lookahead:
            latest_sequence = _load_sequence(working_sequence)
            lookahead_path = task_dir / "downstream_safety_context.json"
            _write_downstream_safety_context(latest_sequence, task_id, lookahead_path)
            cmd.extend(["--downstream-safety-context", str(lookahead_path)])

        print(f"\n=== {ordinal}/{len(task_ids)} {task_id} ===", flush=True)
        print(" ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=REPO_ROOT, check=True)

        summary_path = task_dir / "optimization_summary.json"
        if not summary_path.exists():
            raise SystemExit(f"ERROR: missing task optimizer summary: {summary_path}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        refined = Path(str(summary.get("refined_sequence") or ""))
        if not refined.exists():
            raise SystemExit(f"ERROR: missing refined sequence from task optimizer: {refined}")
        shutil.copy2(refined, working_sequence)

        next_none = str(summary.get("best_none_replay_dir") or "").strip()
        next_self = str(summary.get("best_self_replay_dir") or "").strip()
        if next_none:
            none_run_dir = Path(next_none)
        if next_self and not freeze_self_for_task:
            self_run_dir = Path(next_self)
            if args.freeze_self_evolution_after_upstream and role == "upstream":
                frozen_self_run_dir = self_run_dir
        next_none_stm = str(summary.get("best_none_stm_context_file") or "").strip()
        next_self_stm = str(summary.get("best_self_stm_context_file") or "").strip()
        next_none_session = str(summary.get("best_none_session_state_file") or "").strip()
        next_self_session = str(summary.get("best_self_session_state_file") or "").strip()
        if carry_selected_context and next_none_stm:
            none_stm_context_file = Path(next_none_stm)
        if carry_selected_context and next_self_stm:
            self_stm_context_file = Path(next_self_stm)
        if next_none_session:
            none_session_state_file = Path(next_none_session)
        if next_self_session:
            self_session_state_file = Path(next_self_session)

        _write_jsonl(
            output_dir / "sequence_optimization_log.jsonl",
            {
                "task_id": task_id,
                "task_dir": str(task_dir),
                "accepted_prompt_changed": bool(summary.get("accepted_prompt_changed")),
                "final_score": summary.get("final_score"),
                "final_hard_ok": summary.get("final_hard_ok"),
                "final_reasons": summary.get("final_reasons"),
                "downstream_safety_lookahead": bool(args.downstream_safety_lookahead),
                "freeze_self_evolution_after_upstream": bool(args.freeze_self_evolution_after_upstream),
                "contextual_scoping_defense": bool(args.contextual_scoping_defense),
                "compartmentalize_evolution_updates": bool(args.compartmentalize_evolution_updates),
                "salient_evolution": bool(args.salient_evolution),
                "role": role,
                "max_task_retries": max_task_retries_for_task,
                "fresh_session_boundary": fresh_session_for_task,
                "frozen_self_run_dir": str(frozen_self_run_dir) if frozen_self_run_dir else "",
                "next_none_run_dir": str(none_run_dir) if none_run_dir else "",
                "next_self_run_dir": str(self_run_dir) if self_run_dir else "",
                "carry_selected_context": bool(carry_selected_context),
                "next_none_stm_context_file": str(none_stm_context_file) if none_stm_context_file else "",
                "next_self_stm_context_file": str(self_stm_context_file) if self_stm_context_file else "",
                "next_none_session_state_file": str(none_session_state_file) if none_session_state_file else "",
                "next_self_session_state_file": str(self_session_state_file) if self_session_state_file else "",
            },
        )

    final_sequence = output_dir / "final_sequence.yaml"
    shutil.copy2(working_sequence, final_sequence)
    if args.in_place:
        shutil.copy2(final_sequence, task_sequence)
        print(f"Updated in place: {task_sequence}", flush=True)
    print(f"\nWrote final sequence: {final_sequence}", flush=True)
    print(f"Wrote log: {output_dir / 'sequence_optimization_log.jsonl'}", flush=True)


if __name__ == "__main__":
    started = time.time()
    try:
        main()
    finally:
        print(f"Elapsed: {time.time() - started:.1f}s", flush=True)
