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


def train(args, paths, environment) -> None:
    selector_command("validate", args, paths, environment)

    common = (
        "--base-monitor", SOURCE / "selector_verifier.py",
        "--feature-module", SOURCE / "progression_features.py",
        "--data-root", args.data_root,
        "--split-file", args.split_file,
        "--output-dir", paths["progression"],
    )
    execute("progression_filter.py", *common, "--stage", "select", environment=environment)
    execute("progression_filter.py", *common, "--stage", "validate", environment=environment)

    execute(
        "combine.py",
        "--local-scores", paths["progression"] / "validation_predictions.jsonl",
        "--verifier-scores", paths["selector"] / "validation_raw_scores.jsonl",
        "--output-dir", paths["final"],
        "--stage", "select",
        environment=environment,
    )

    validation_manifest = read_json(paths["selector"] / "validation_manifest.json")
    frozen = {
        key: validation_manifest[key]
        for key in ("config_sha256", "prompt_sha256", "split_sha256")
    }
    frozen["selection"] = "validation complete; monitor frozen before test"
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
    args = parser.parse_args()
    args.data_root = args.data_root.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.split_file = (
        args.split_file.resolve() if args.split_file else args.output_dir / "split.json"
    )
    if not args.data_root.is_dir():
        raise FileNotFoundError(f"data root does not exist: {args.data_root}")

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
