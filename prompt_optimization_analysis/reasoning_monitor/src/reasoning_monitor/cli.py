"""Run the complete training, validation, and test pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


SOURCE = Path(__file__).resolve().parent


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def execute(script: str, *arguments: str, environment: dict[str, str]) -> None:
    command = [sys.executable, str(SOURCE / script), *map(str, arguments)]
    subprocess.run(command, check=True, env=environment)


def selector_command(stage: str, args, paths, environment) -> None:
    execute(
        "selector_verifier.py",
        "--data-root", args.data_root,
        "--split-file", args.split_file,
        "--output-dir", paths["selector"],
        "--stage", stage,
        environment=environment,
    )


def prepare(args, paths, environment) -> None:
    selector_command("prepare", args, paths, environment)


def combine_validation(paths, environment) -> dict:
    execute(
        "combine.py",
        "--local-scores", paths["progression"] / "validation_predictions.jsonl",
        "--verifier-scores", paths["selector"] / "validation_raw_scores.jsonl",
        "--output-dir", paths["final"],
        "--stage", "select",
        environment=environment,
    )
    return read_json(paths["final"] / "selection_and_report.json")


def prompt_rank(report: dict) -> tuple[float, float, float, float]:
    metrics = report["metrics"]
    recalls = [
        metrics["by_surface"][surface]["failure_recall"]
        for surface in ("controller", "memory", "tools_skills")
    ]
    overall = metrics["overall"]
    return (
        overall["accuracy"],
        overall["failure_recall"],
        min(recalls),
        -overall["false_positive_rate"],
    )


def make_validation_feedback(paths) -> dict:
    """Describe semantic validation errors without trace identifiers or raw traces."""
    selection = read_json(paths["final"] / "selection_and_report.json")
    decisions = read_jsonl(paths["final"] / "validation_predictions.jsonl")
    verifier_rows = {
        row["trace_id"]: row
        for row in read_jsonl(paths["selector"] / "validation_raw_scores.jsonl")
    }
    errors = []
    excluded_selector_or_filter_misses = 0
    for decision in decisions:
        failed = decision["outcome"] == "failure"
        predicted = bool(decision["prediction"])
        if failed == predicted:
            continue
        verifier_row = verifier_rows[decision["trace_id"]]
        verdict = verifier_row.get("verifier")
        local_threshold = float(
            selection["surface_thresholds"][decision["surface"]]["local_threshold"]
        )
        local_passed = float(decision["local_score"]) >= local_threshold
        if failed and (not local_passed or not isinstance(verdict, dict)):
            excluded_selector_or_filter_misses += 1
            continue
        verdict = verdict if isinstance(verdict, dict) else {}
        errors.append({
            "error": "false_negative" if failed else "false_positive",
            "verifier_score": float(decision["verifier_score"]),
            "harm_type": verdict.get("harm_type", "none"),
            "final_state": verdict.get("final_state", "unclear"),
            "risk_object": verdict.get("risk_object", ""),
            "boundary_or_requirement": verdict.get("boundary_or_requirement", ""),
            "reason": verdict.get("reason", ""),
            "used_full_trace_fallback": bool(
                verifier_row.get("full_trace_fallback", False)
            ),
        })
    return {
        "instruction": (
            "Revise only general semantic decision rules that address these validation "
            "errors. Do not copy names, values, or task-specific language into the prompt."
        ),
        "validation_metrics": selection["metrics"],
        "semantic_errors": errors,
        "errors_not_addressable_by_the_verifier_prompt": (
            excluded_selector_or_filter_misses
        ),
        "test_data_used": False,
    }


def evaluate_prompt(prompt: Path, args, paths, environment) -> dict:
    environment["REASONING_MONITOR_PROMPT"] = str(prompt.resolve())
    selector_command("validate", args, paths, environment)
    return combine_validation(paths, environment)


def refine_verifier_prompt(args, paths, environment) -> Path:
    search = paths["root"] / "prompt_refinement"
    search.mkdir(parents=True, exist_ok=True)
    initial = Path(
        environment.get("REASONING_MONITOR_PROMPT", SOURCE / "verifier_prompt.md")
    ).resolve()
    current = search / "prompt_00.md"
    current.write_text(initial.read_text(encoding="utf-8"), encoding="utf-8")
    config = read_json(SOURCE / "config.json")
    writer_model = (
        args.prompt_writer_model
        or environment.get("REASONING_MONITOR_MODEL")
        or str(config["verifier_model"])
    )
    candidates = []
    for round_index in range(args.prompt_refinement_rounds + 1):
        report = evaluate_prompt(current, args, paths, environment)
        candidates.append({
            "round": round_index,
            "prompt": current,
            "prompt_sha256": sha256(current),
            "report": report,
        })
        if round_index == args.prompt_refinement_rounds:
            break
        feedback = make_validation_feedback(paths)
        feedback_path = search / f"feedback_{round_index:02d}.json"
        write_json(feedback_path, feedback)
        if not feedback["semantic_errors"]:
            break
        revised = search / f"prompt_{round_index + 1:02d}.md"
        execute(
            "derive_verifier_prompt.py",
            "--model", writer_model,
            "--seed-prompt", current,
            "--validation-feedback", feedback_path,
            "--output-prompt", revised,
            environment=environment,
        )
        current = revised

    selected = max(candidates, key=lambda row: prompt_rank(row["report"]))
    selected_prompt = paths["root"] / "verifier_prompt.md"
    selected_prompt.write_text(
        selected["prompt"].read_text(encoding="utf-8"), encoding="utf-8"
    )
    write_json(paths["root"] / "prompt_refinement_report.json", {
        "selection_rule": (
            "maximize accuracy, then failure recall and minimum per-surface "
            "recall, subject to the monitor FPR constraint"
        ),
        "selected_round": selected["round"],
        "selected_prompt_sha256": sha256(selected_prompt),
        "rounds": [
            {
                "round": row["round"],
                "prompt_sha256": row["prompt_sha256"],
                "metrics": row["report"]["metrics"],
            }
            for row in candidates
        ],
        "test_data_used": False,
    })
    # Recreate the canonical validation artifacts with the selected prompt.
    evaluate_prompt(selected_prompt, args, paths, environment)
    return selected_prompt


def train(args, paths, environment) -> None:
    # Ensure the grouped split exists before either learned component is fitted.
    prepare(args, paths, environment)
    common = (
        "--base-monitor", SOURCE / "selector_verifier.py",
        "--feature-module", SOURCE / "progression_features.py",
        "--data-root", args.data_root,
        "--split-file", args.split_file,
        "--output-dir", paths["progression"],
    )
    execute("progression_filter.py", *common, "--stage", "select", environment=environment)
    execute("progression_filter.py", *common, "--stage", "validate", environment=environment)
    selected_prompt = refine_verifier_prompt(args, paths, environment)

    validation_manifest = read_json(paths["selector"] / "validation_manifest.json")
    frozen = {
        key: validation_manifest[key]
        for key in ("config_sha256", "prompt_sha256", "split_sha256")
    }
    frozen["selection"] = "training and validation complete; monitor frozen before test"
    write_json(paths["selector"] / "FROZEN_FOR_TEST.json", frozen)
    write_json(paths["root"] / "TRAINED_MONITOR.json", {
        "architecture": [
            "ExtraTrees progression filter",
            "multiple-instance progression selector",
            "API-served semantic verifier",
        ],
        "split_sha256": sha256(args.split_file),
        "progression_model_sha256": sha256(paths["progression"] / "monitor.joblib"),
        "selector_model_sha256": sha256(paths["selector"] / "stage1.joblib"),
        "verifier_prompt_sha256": sha256(selected_prompt),
        "thresholds_sha256": sha256(paths["final"] / "selection_and_report.json"),
        "test_evaluated": False,
    })


def test(args, paths, environment) -> None:
    required = [
        paths["selector"] / "FROZEN_FOR_TEST.json",
        paths["progression"] / "monitor.joblib",
        paths["final"] / "selection_and_report.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError("Train and freeze the monitor first; missing: " + ", ".join(missing))

    selector_command("test", args, paths, environment)
    execute(
        "progression_filter.py",
        "--base-monitor", SOURCE / "selector_verifier.py",
        "--feature-module", SOURCE / "progression_features.py",
        "--data-root", args.data_root,
        "--split-file", args.split_file,
        "--output-dir", paths["progression"],
        "--stage", "test",
        environment=environment,
    )
    execute(
        "combine.py",
        "--local-scores", paths["progression"] / "test_predictions.jsonl",
        "--verifier-scores", paths["selector"] / "test_predictions.jsonl",
        "--output-dir", paths["final"],
        "--stage", "test",
        "--selection", paths["final"] / "selection_and_report.json",
        environment=environment,
    )
    manifest = read_json(paths["root"] / "TRAINED_MONITOR.json")
    manifest["test_evaluated"] = True
    manifest["test_report_sha256"] = sha256(paths["final"] / "test_report.json")
    write_json(paths["root"] / "TRAINED_MONITOR.json", manifest)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the reasoning-trace monitor without Codex."
    )
    parser.add_argument("command", choices=("prepare", "train", "test", "run"))
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--split-file", type=Path,
        help="Frozen grouped split; defaults to OUTPUT_DIR/split.json",
    )
    parser.add_argument(
        "--verifier-model",
        help="API model name; defaults to the model recorded in config.json",
    )
    parser.add_argument(
        "--verifier-prompt", type=Path,
        help="Derived verifier prompt; defaults to the included fixed prompt",
    )
    parser.add_argument(
        "--prompt-writer-model",
        help="API model used to revise the verifier prompt from validation errors",
    )
    parser.add_argument(
        "--prompt-refinement-rounds", type=int, default=2,
        help="Maximum validation-based verifier-prompt revisions (default: 2)",
    )
    args = parser.parse_args()
    args.data_root = args.data_root.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.split_file = (
        args.split_file.resolve() if args.split_file else args.output_dir / "split.json"
    )
    if not args.data_root.is_dir():
        raise FileNotFoundError(f"data root does not exist: {args.data_root}")
    if args.prompt_refinement_rounds < 0:
        raise ValueError("--prompt-refinement-rounds must be nonnegative")

    environment = os.environ.copy()
    if args.verifier_model:
        environment["REASONING_MONITOR_MODEL"] = args.verifier_model
    if args.verifier_prompt:
        prompt = args.verifier_prompt.resolve()
        if not prompt.is_file():
            raise FileNotFoundError(f"verifier prompt does not exist: {prompt}")
        environment["REASONING_MONITOR_PROMPT"] = str(prompt)
    if args.command in {"train", "test", "run"} and not environment.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required for verifier calls")

    paths = {
        "root": args.output_dir,
        "selector": args.output_dir / "selector_verifier",
        "progression": args.output_dir / "progression_filter",
        "final": args.output_dir / "final",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)

    selected_prompt = args.output_dir / "verifier_prompt.md"
    if (
        args.command == "test"
        and not args.verifier_prompt
        and selected_prompt.is_file()
    ):
        environment["REASONING_MONITOR_PROMPT"] = str(selected_prompt)

    if args.command == "prepare":
        prepare(args, paths, environment)
    elif args.command == "train":
        train(args, paths, environment)
    elif args.command == "test":
        test(args, paths, environment)
    else:
        prepare(args, paths, environment)
        train(args, paths, environment)
        test(args, paths, environment)


if __name__ == "__main__":
    main()
