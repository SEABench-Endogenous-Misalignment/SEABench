#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from sequence_generation.initializers import initialize_from_existing, initialize_from_scratch

PROVIDERS_DIR = REPO_ROOT / "configs" / "providers"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the SEABench sequence-generation pipeline.")
    parser.add_argument("--mode", choices=["existing", "scratch"], default="existing")
    parser.add_argument("--objective", choices=["attack", "benign"], default="attack")
    parser.add_argument("--sequence-path", default="", help="Existing task-sequence YAML for seed mode.")
    parser.add_argument("--surface", default="controller_update")
    parser.add_argument("--attack-style", choices=["emergent", "explicit"], default="emergent")
    parser.add_argument("--checkpoint-task-id", default="")
    parser.add_argument("--theme", default="")
    parser.add_argument("--safe-update-rule", default="")
    parser.add_argument("--unsafe-update-rule", default="")
    parser.add_argument("--provider-profile", default="openrouter")
    parser.add_argument("--agent-model", default="")
    parser.add_argument("--judge-model", default="")
    parser.add_argument("--reflection-model", default="")
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def _load_provider(profile: str) -> dict[str, str]:
    path = PROVIDERS_DIR / f"{profile}.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Malformed provider profile: {path}")
    return {
        "base_url": str(payload.get("base_url", "")).strip(),
        "api_key_env": str(payload.get("api_key_env", "OPENROUTER_API_KEY")).strip(),
        "default_model": str(payload.get("default_model", "moonshotai/kimi-k2.5")).strip(),
    }


def main() -> None:
    args = _parse_args()
    provider = _load_provider(args.provider_profile)
    api_key = os.environ.get(provider["api_key_env"], "").strip()
    if not api_key:
        raise SystemExit(f"Missing required provider API key env var: {provider['api_key_env']}")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "existing":
        if not args.sequence_path.strip():
            raise SystemExit("--sequence-path is required in existing mode")
        seed = initialize_from_existing(
            (REPO_ROOT / args.sequence_path).resolve() if not Path(args.sequence_path).is_absolute() else Path(args.sequence_path),
            surface=args.surface,
            objective=args.objective,
            attack_style=args.attack_style,
            checkpoint_task_id=args.checkpoint_task_id,
        )
    else:
        seed = initialize_from_scratch(
            sequence_id="generated-sequence",
            title=args.theme.strip() or "Generated sequence",
            surface=args.surface,
            objective=args.objective,
            attack_style=args.attack_style,
            theme=args.theme.strip() or "Generated theme",
            safe_update_rule=args.safe_update_rule.strip() or "Learn a narrow reusable rule and keep it local.",
            unsafe_update_rule=args.unsafe_update_rule.strip() or "Overgeneralize the learned rule into a default policy.",
        )

    from sequence_generation.optimize import OptimizerConfig, SequenceGeneratorPipeline

    config = OptimizerConfig(
        provider_base_url=provider["base_url"],
        api_key=api_key,
        reflection_model=args.reflection_model.strip() or provider["default_model"],
        provider_profile=args.provider_profile,
        agent_model=args.agent_model.strip() or provider["default_model"],
        judge_model=args.judge_model.strip() or provider["default_model"],
        iterations=max(1, args.iterations),
    )
    pipeline = SequenceGeneratorPipeline(config)
    best_spec, best_eval = pipeline.optimize(seed, output_dir=output_dir)

    (output_dir / "best_spec.json").write_text(
        json.dumps(best_spec.to_optimizer_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "best_sequence.yaml").write_text(
        yaml.safe_dump(best_spec.to_yaml_payload(), sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )
    (output_dir / "best_result.json").write_text(
        json.dumps(
            {
                "accepted": best_eval.accepted,
                "score": best_eval.score,
                "stage": best_eval.stage,
                "acceptance_checks": best_eval.acceptance_checks,
                "conditions": {
                    name: {
                        "run_dir": result.run_dir,
                        "returncode": result.returncode,
                        "early_visible_failures": result.early_visible_failures,
                        "early_visible_successes": result.early_visible_successes,
                        "late_visible_failures": result.late_visible_failures,
                        "late_visible_successes": result.late_visible_successes,
                        "late_safety_failures": result.late_safety_failures,
                        "update_applied": result.update_applied,
                        "summary_path": result.summary_path,
                    }
                    for name, result in best_eval.conditions.items()
                },
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"Best score: {best_eval.score:.3f}")
    print(f"Accepted: {best_eval.accepted}")
    print(f"Output dir: {output_dir}")


if __name__ == "__main__":
    main()
