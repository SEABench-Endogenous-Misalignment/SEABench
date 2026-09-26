#!/usr/bin/env python3
"""Extract selected-candidate reasoning traces from a cascade optimization run into markdown.

For each numbered task step under `<output-dir>/cascade_optimization/`, this looks up
which candidate was selected (from `optimization_summary.json`, falling back to
`optimization_log.jsonl` if that's missing) and pulls the reasoning trace for the last
attempt of that candidate's replay, for both the `none` (non-evolving control) and
`self_evolving` arms. It writes one markdown file per arm:

  <output-dir>/analysis/selected_candidate_traces_none.md
  <output-dir>/analysis/selected_candidate_traces_self_evolving.md

Each file contains, per task: the task prompt, the final answer (from the last retry
attempt), and the model's reasoning trace for every turn of that attempt.

Usage:
  python3 prompt_optimization_analysis/extract_reasoning_traces.py \\
    --output-dir prompt_optimization/out/cot_tests/kimi_professional_activities_hallucination_salient_evolution
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ARMS = ("none", "self_evolving")

ARM_REPLAY_KEY = {"none": "best_none_replay_dir", "self_evolving": "best_self_replay_dir"}

ARM_DESCRIPTION = {
    "none": (
        "Non-evolving (none surface) agent run, selected candidate, per task. Task prompt, "
        "final answer (last attempt), and reasoning traces for each turn."
    ),
    "self_evolving": (
        "Self-evolving agent run, selected candidate, per task. Task prompt, final answer "
        "(last attempt), and reasoning traces for each turn."
    ),
}

ARM_TITLE_SUFFIX = {"none": " (none / non-evolving)", "self_evolving": ""}

SEP = "\n\n\n---\n\n"


@dataclass
class TaskTrace:
    task_label: str
    role: str
    selected_candidate: str
    replay_dir: str
    task_prompt: str
    answer: str
    attempt_number: int
    total_attempts: int
    passed: Any
    turns: list[dict[str, Any]] = field(default_factory=list)


def _numbered_task_dirs(cascade_dir: Path) -> list[Path]:
    dirs = [d for d in cascade_dir.iterdir() if d.is_dir() and re.match(r"^\d+_task_\d+$", d.name)]
    return sorted(dirs, key=lambda d: int(d.name.split("_")[0]))


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _selected_candidate_fallback(task_dir: Path) -> str:
    log_path = task_dir / "optimization_log.jsonl"
    if not log_path.exists():
        return "current"
    selected = "current"
    for line in log_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if entry.get("selected") and entry.get("candidate"):
            selected = str(entry["candidate"])
    return selected


def _glob_replay_dir(task_dir: Path, candidate: str, arm: str) -> Path | None:
    pattern = str(task_dir / "evaluations" / candidate / arm / "*")
    matches = sorted(p for p in glob.glob(pattern) if (Path(p) / "task_result.json").exists())
    return Path(matches[-1]) if matches else None


def _resolve_task(task_dir: Path, summary: dict[str, Any] | None, arm: str) -> tuple[str, str, Path | None]:
    """Returns (role, selected_candidate, replay_dir)."""
    if summary is not None:
        role = str(summary.get("role") or "")
        candidate = str(summary.get("selected_candidate") or "current")
        replay_dir_str = summary.get(ARM_REPLAY_KEY[arm])
        if replay_dir_str and Path(replay_dir_str).exists():
            return role, candidate, Path(replay_dir_str)
    else:
        role = ""
        candidate = _selected_candidate_fallback(task_dir)
    return role, candidate, _glob_replay_dir(task_dir, candidate, arm)


def collect_traces(output_dir: Path, arm: str) -> list[TaskTrace]:
    cascade_dir = output_dir / "cascade_optimization"
    if not cascade_dir.exists():
        raise SystemExit(f"No cascade_optimization/ directory found under {output_dir}")

    traces: list[TaskTrace] = []
    for task_dir in _numbered_task_dirs(cascade_dir):
        summary = _load_json(task_dir / "optimization_summary.json")
        role, candidate, replay_dir = _resolve_task(task_dir, summary, arm)
        if replay_dir is None:
            print(f"  [skip] {task_dir.name} ({arm}): no replay dir found for candidate {candidate!r}")
            continue

        result = _load_json(replay_dir / "task_result.json")
        if result is None:
            print(f"  [skip] {task_dir.name} ({arm}): no task_result.json in {replay_dir}")
            continue

        attempts = result.get("attempts")
        if not isinstance(attempts, list) or not attempts:
            print(f"  [skip] {task_dir.name} ({arm}): no attempts recorded")
            continue
        last_attempt = attempts[-1]

        task_label = str(result.get("task_id") or task_dir.name.split("_", 1)[-1])
        task_prompt = str(result.get("task_prompt") or last_attempt.get("task_prompt") or "")
        answer = str(result.get("answer") or last_attempt.get("answer") or "")
        passed = last_attempt.get("passed", result.get("passed"))
        turns = list(((last_attempt.get("task_run") or {}).get("reasoning_traces")) or [])

        traces.append(
            TaskTrace(
                task_label=task_label,
                role=role,
                selected_candidate=candidate,
                replay_dir=str(replay_dir),
                task_prompt=task_prompt,
                answer=answer,
                attempt_number=int(last_attempt.get("attempt") or len(attempts)),
                total_attempts=len(attempts),
                passed=passed,
                turns=turns,
            )
        )
    return traces


def _render_task_block(trace: TaskTrace) -> str:
    heading = f"## {trace.task_label} ({trace.role})" if trace.role else f"## {trace.task_label}"
    lines = [
        heading,
        "",
        f"**Selected candidate:** `{trace.selected_candidate}`  ",
        f"**Attempt shown:** {trace.attempt_number} of {trace.total_attempts} (passed: {trace.passed})  ",
        f"**Replay dir:** `{trace.replay_dir}`",
        "",
        "### Task Prompt",
        "",
        "```",
        trace.task_prompt.strip("\n"),
        "```",
        "",
        "### Model Response (final answer)",
        "",
        "```",
        trace.answer.strip("\n"),
        "```",
        "",
        "### Reasoning Traces",
        "",
    ]
    if trace.turns:
        for i, turn in enumerate(trace.turns):
            turn_num = turn.get("turn", i)
            reasoning = str(turn.get("reasoning") or "").strip()
            lines.append(f"**Turn {turn_num}:**")
            lines.append("")
            lines.append("```")
            lines.append(reasoning)
            lines.append("```")
            if i < len(trace.turns) - 1:
                lines.append("")
    else:
        lines.append("_No reasoning traces recorded for this attempt._")
    return "\n".join(lines)


def render_markdown(output_dir: Path, arm: str, traces: list[TaskTrace]) -> str:
    title_base = output_dir.name.replace("_", " ").replace("-", " ").title()
    title = f"# {title_base} — Cascade Optimization Traces{ARM_TITLE_SUFFIX[arm]}"
    description = ARM_DESCRIPTION[arm]
    blocks = [f"{title}\n\n{description}"] + [_render_task_block(t) for t in traces]
    return SEP.join(blocks) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", required=True, help="Cascade run root, e.g. prompt_optimization/out/<name>")
    parser.add_argument(
        "--analysis-dir", default="", help="Defaults to <output-dir>/analysis/reasoning_traces_consolidated"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    analysis_dir = (
        Path(args.analysis_dir).resolve()
        if args.analysis_dir
        else output_dir / "analysis" / "reasoning_traces_consolidated"
    )
    analysis_dir.mkdir(parents=True, exist_ok=True)

    for arm in ARMS:
        print(f"Collecting {arm} traces from {output_dir}...")
        traces = collect_traces(output_dir, arm)
        if not traces:
            raise SystemExit(f"No reasoning traces found for arm={arm!r} under {output_dir}")
        markdown = render_markdown(output_dir, arm, traces)
        out_path = analysis_dir / f"selected_candidate_traces_{arm}.md"
        out_path.write_text(markdown, encoding="utf-8")
        print(f"Wrote {out_path} ({len(traces)} tasks)")


if __name__ == "__main__":
    main()
