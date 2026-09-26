#!/usr/bin/env python3
"""
Entry point for the personal assistant sandbox.

Inside the Apptainer container this file is executed from /repo/scripts/run.py.
Run via:  ./scripts/run.sh          (recommended)
          python3 scripts/run.py
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import re
import shutil
from pathlib import Path

_REPO_ROOT = Path(os.environ.get("REPO_ROOT", Path(__file__).resolve().parent.parent))

# Ensure the repo root is on the path.
sys.path.insert(0, str(_REPO_ROOT))

from agent.loop import AgentLoop
from agent.session import Session
from runtime.config import Config
from runtime.evolution import validate_evolution_surface, write_evolution_controller, allowed_extension_files as get_allowed_extension_files
from runtime.run_manifest import apply_run_manifest, load_run_manifest
from runtime.task_loader import load_yaml_task_suite, load_yaml_task_suite_from_files
from runtime.validation_dispatcher import initialize_validation_dispatcher
from analysis_tools.self_improvement.tracker import EvolutionTracker
from sandbox.workspace import SandboxGuard, reset_workspace_from_immutable_sources
from agent.tools.core.compaction import make_compaction_tool
from agent.tools.core.created_tools import load_created_tools
from agent.tools.core.filesystem import make_filesystem_tools
from agent.tools.core.memory import make_memory_tools
from agent.tools.core.self_improve import make_self_improve_tools
from agent.tools.registry import ToolRegistry

_CONTROLLER_SURFACE = _REPO_ROOT / "agent" / "seeded_files"
_BENIGN_TASKS_DIR = _REPO_ROOT / "tasks" / "benign"
_RUN_CONFIGS_DIR = _REPO_ROOT / "configs" / "runs"
_PROVIDERS_DIR = _REPO_ROOT / "configs" / "providers"


def _task_role(task) -> str:
    role = str(getattr(task, "optimization_role", "") or "").strip().lower()
    if role in {"upstream", "downstream"}:
        return role
    validator = getattr(task, "validator_spec", {}) or {}
    if isinstance(validator, dict) and isinstance(validator.get("safety_llm"), dict):
        return "downstream"
    return "upstream"


class RoleAwareAgentLoop(AgentLoop):
    def __init__(
        self,
        *args,
        base_surface: str,
        upstream_max_task_retries: int,
        downstream_max_task_retries: int,
        freeze_self_evolution_after_upstream: bool,
        downstream_standalone: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.base_surface = base_surface
        self.upstream_max_task_retries = max(0, int(upstream_max_task_retries))
        self.downstream_max_task_retries = max(0, int(downstream_max_task_retries))
        self.freeze_self_evolution_after_upstream = bool(freeze_self_evolution_after_upstream)
        self.downstream_standalone = bool(downstream_standalone)

    def _configure_task_mode(self, task) -> None:
        role = _task_role(task)
        if role == "downstream":
            retries = self.downstream_max_task_retries
            surface = "none" if self.freeze_self_evolution_after_upstream else self.base_surface
        else:
            retries = self.upstream_max_task_retries
            surface = self.base_surface
        self.session.config.max_task_retries = retries
        self.session.config.evolution_surface = surface
        write_evolution_controller(
            self.workspace_dir,
            surface,
            contextual_scoping_defense=self.session.config.contextual_scoping_defense,
            compartmentalize_evolution_updates=self.session.config.compartmentalize_evolution_updates,
            salient_evolution=self.session.config.salient_evolution,
        )
        self.tracker.log(
            "task_mode_configured",
            task_id=task.task_id,
            role=role,
            evolution_surface=surface,
            max_task_retries=retries,
            downstream_frozen=bool(role == "downstream" and self.freeze_self_evolution_after_upstream),
        )

    def _sequence_tool_limits(self, task):
        allow, deny = super()._sequence_tool_limits(task)
        if _task_role(task) == "downstream" and self.freeze_self_evolution_after_upstream:
            deny_set = set(deny or [])
            deny_set.update({"record_learning", "apply_controller_update", "update_memory", "create_skill", "create_tool"})
            deny = sorted(deny_set)
        return allow, deny

    def _should_reset_session_before_task(self, previous_task, task) -> bool:
        if self.downstream_standalone and _task_role(task) == "downstream":
            return True
        if not self.freeze_self_evolution_after_upstream or previous_task is None:
            return False
        return _task_role(previous_task) == "upstream" and _task_role(task) == "downstream"

    def _run_task(self, task, handoff: str = "") -> str:
        self._configure_task_mode(task)
        # OpenClaw-style carryover should come from the restored conversation /
        # session state, not an extra explicit prior-task handoff string.
        super()._run_task(task, "")
        return ""


def _safe_slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return slug or "run"


def _task_sequence_slug(task_files: list[Path]) -> str:
    if not task_files:
        return "no-tasks"
    stems = [_safe_slug(path.stem) for path in task_files[:3]]
    if len(task_files) > 3:
        stems.append(f"plus-{len(task_files) - 3}")
    return "-".join(stems)


def _model_slug(value: str) -> str:
    slug = _safe_slug(value)
    return slug[:48] if len(slug) > 48 else slug


def _resolve_path_arg(value: str, repo_root: Path) -> Path:
    raw = Path(value)
    if raw.is_absolute():
        return raw
    return (repo_root / raw).resolve()


def _find_latest_incomplete_run(
    artifacts_root: Path,
    *,
    manifest_slug: str,
    surface_slug: str,
    model_slug: str,
    task_slug: str,
) -> Path | None:
    prefix = f"run_{manifest_slug}__{surface_slug}__{model_slug}__{task_slug}__"
    candidates = sorted(
        [p for p in artifacts_root.glob(f"{prefix}*") if p.is_dir()],
        reverse=True,
    )
    for run_dir in candidates:
        summary_path = run_dir / "checkpoints" / "run_summary.json"
        if not summary_path.exists():
            return run_dir
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            return run_dir
        completed = int(payload.get("tasks_completed", 0) or 0)
        total = int(payload.get("tasks_total", 0) or 0)
        if total <= 0 or completed < total:
            return run_dir
    return None


def _surface_slug_from_run_dir(run_dir: Path) -> str | None:
    name = run_dir.name
    if not name.startswith("run_"):
        return None
    parts = name.split("__")
    if len(parts) < 3:
        return None
    return parts[1]


def _load_resume_state(
    run_dir: Path,
    task_suite: list,
) -> tuple[list[dict], int, dict]:
    summary_path = run_dir / "checkpoints" / "run_summary.json"
    if not summary_path.exists():
        return [], 0, {}

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    prior_results = payload.get("results", [])
    if not isinstance(prior_results, list):
        return [], 0, {}

    kept_results: list[dict] = []
    for idx, task in enumerate(task_suite):
        if idx >= len(prior_results):
            return kept_results, idx, _resume_metadata(run_dir, task_suite, kept_results, idx)
        result = prior_results[idx]
        if not isinstance(result, dict) or str(result.get("task_id", "")) != task.task_id:
            return kept_results, idx, _resume_metadata(run_dir, task_suite, kept_results, idx)
        kept_results.append(result)

    return kept_results, len(task_suite), _resume_metadata(run_dir, task_suite, kept_results, len(task_suite))


def _resume_metadata(
    run_dir: Path,
    task_suite: list,
    kept_results: list[dict],
    start_task_index: int,
) -> dict:
    if not kept_results:
        return {}

    last_result = kept_results[-1]
    next_task = task_suite[start_task_index] if start_task_index < len(task_suite) else None
    last_sequence_id = str(last_result.get("sequence_id") or "")
    next_sequence_id = str(getattr(next_task, "sequence_id", "") or "") if next_task is not None else ""
    same_sequence = bool(next_task is not None and last_sequence_id and last_sequence_id == next_sequence_id)

    checkpoint_workspace = ""
    session_state_path = ""
    attempts = last_result.get("attempts") if isinstance(last_result, dict) else None
    if isinstance(attempts, list) and attempts:
        last_attempt = attempts[-1]
        if isinstance(last_attempt, dict):
            checkpoint = last_attempt.get("evolved_agent_checkpoint")
            if isinstance(checkpoint, dict):
                checkpoint_workspace = str(checkpoint.get("workspace_root") or "")
    session_state_checkpoint = last_result.get("session_state_checkpoint") if isinstance(last_result, dict) else None
    if isinstance(session_state_checkpoint, dict):
        session_state_path = str(session_state_checkpoint.get("session_state_path") or "")
    continuation = last_result.get("continuation_state") if isinstance(last_result, dict) else None
    exact_handoff = ""
    if isinstance(continuation, dict):
        exact_handoff = str(continuation.get("next_task_handoff") or "").strip()

    return {
        "last_task_id": str(last_result.get("task_id") or ""),
        "last_sequence_id": last_sequence_id,
        "same_sequence": same_sequence,
        "checkpoint_workspace": checkpoint_workspace,
        "session_state_path": session_state_path,
        "handoff": exact_handoff or (
            f"Task {last_result.get('task_id')} ({last_result.get('title', '')}) "
            f"{'passed' if last_result.get('passed') else 'failed' if last_result.get('passed') is False else 'unscored'}. "
            f"Validator: {last_result.get('feedback', '')}"
        ),
    }


def _restore_workspace_checkpoint(
    *,
    run_dir: Path,
    workspace_dir: Path,
    checkpoint_workspace: str,
) -> bool:
    if not checkpoint_workspace:
        return False
    src_root = run_dir / checkpoint_workspace
    if not src_root.exists() or not src_root.is_dir():
        return False
    for src in src_root.rglob("*"):
        if not src.is_file():
            continue
        rel = src.relative_to(src_root)
        dst = workspace_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    return True


def _restore_session_state(
    *,
    run_dir: Path,
    session: Session,
    session_state_path: str,
) -> bool:
    if not session_state_path:
        return False
    src = run_dir / session_state_path
    if not src.exists() or not src.is_file():
        return False
    payload = json.loads(src.read_text(encoding="utf-8"))
    session.restore_state(payload)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SEABench task sequences.")
    parser.add_argument(
        "--run-config",
        default=str(_RUN_CONFIGS_DIR / "controller_update_computer_use_privacy.yaml"),
        help="Path to a run-config YAML file.",
    )
    parser.add_argument(
        "--resume-run-dir",
        default="",
        help="Resume a previous run directory by restarting from the first incomplete task.",
    )
    parser.add_argument(
        "--resume-latest",
        action="store_true",
        help="Resume the latest incomplete run matching this run config, if one exists.",
    )
    parser.add_argument(
        "--downstream-standalone",
        action="store_true",
        default=False,
        help="Reset session context before each downstream task, so each runs with a fresh conversation history (workspace/controller files are preserved).",
    )
    parser.add_argument(
        "--run-baseline-upstream",
        action="store_true",
        default=False,
        help="When surface is 'none', also run upstream tasks. By default, upstream tasks are skipped for the none surface.",
    )
    args = parser.parse_args()

    config = Config()
    run_manifest = load_run_manifest(Path(args.run_config), _REPO_ROOT)
    config = apply_run_manifest(config, run_manifest, _PROVIDERS_DIR)
    config.evolution_surface = validate_evolution_surface(config.evolution_surface)

    artifacts_root = Path(config.artifacts_dir)
    env_assets_dir = Path(config.env_assets_dir)
    artifacts_root.mkdir(parents=True, exist_ok=True)

    manifest_slug = _safe_slug(run_manifest.name)
    surface_slug = _safe_slug(config.evolution_surface)
    agent_model_slug = _model_slug(config.model)
    task_slug = _task_sequence_slug(run_manifest.task_files)
    resume_run_dir: Path | None = None
    if args.resume_run_dir:
        resume_run_dir = _resolve_path_arg(args.resume_run_dir, _REPO_ROOT)
    elif args.resume_latest:
        resume_run_dir = _find_latest_incomplete_run(
            artifacts_root,
            manifest_slug=manifest_slug,
            surface_slug=surface_slug,
            model_slug=agent_model_slug,
            task_slug=task_slug,
        )

    if resume_run_dir is not None:
        resume_surface_slug = _surface_slug_from_run_dir(resume_run_dir)
        if resume_surface_slug and resume_surface_slug != surface_slug:
            raise SystemExit(
                "ERROR: resume run directory surface does not match run config.\n"
                f"  run config evolution_surface: {config.evolution_surface} ({surface_slug})\n"
                f"  resume directory surface:   {resume_surface_slug}\n"
                "Use the matching --surface in the experiment runner, or pass a run config "
                "with the same evolution_surface as the artifact directory."
            )

    # Load task suite early so we can check resume state before committing to a run_dir.
    task_suite = (
        load_yaml_task_suite_from_files(run_manifest.task_files)
        if run_manifest.task_files
        else load_yaml_task_suite(_BENIGN_TASKS_DIR)
    )
    if config.evolution_surface == "none" and not args.run_baseline_upstream:
        task_suite = [t for t in task_suite if _task_role(t) == "downstream"]

    initial_results: list[dict] = []
    start_task_index = 0
    resume_meta: dict = {}
    if resume_run_dir is not None:
        initial_results, start_task_index, resume_meta = _load_resume_state(resume_run_dir, task_suite)
        if start_task_index >= len(task_suite):
            print(f"Resuming existing run directory: {resume_run_dir}")
            print("Run already complete; nothing to resume.")
            return
        if initial_results:
            run_dir = resume_run_dir
            print(f"Resuming existing run directory: {run_dir}")
            resumed_task_ids = [str(item.get("task_id", "")) for item in initial_results]
            print(
                f"Resume point: restarting from task index {start_task_index} "
                f"after {len(initial_results)} completed task(s)."
            )
            print(f"Completed prefix kept: {', '.join(resumed_task_ids)}")
        else:
            print(
                f"Resume point: no completed task prefix found in {resume_run_dir}; "
                "leaving it untouched and starting a new run directory."
            )
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            run_id = f"run_{manifest_slug}__{surface_slug}__{task_slug}__{timestamp}"
            run_dir = artifacts_root / run_id
            print(f"Starting new run directory: {run_dir}")
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"run_{manifest_slug}__{surface_slug}__{agent_model_slug}__{task_slug}__{timestamp}"
        run_dir = artifacts_root / run_id
        print(f"Starting new run directory: {run_dir}")

    workspace_dir = Path(config.workspace_dir) if config.workspace_dir else run_dir / "workspace"
    log_dir = run_dir / "logs" / "event_logs"
    checkpoints_dir = run_dir / "checkpoints"
    analysis_dir = run_dir / "analysis"
    for path in (workspace_dir, log_dir, checkpoints_dir, analysis_dir):
        path.mkdir(parents=True, exist_ok=True)

    # Fresh workspace: env files + controller surface files seeded in
    print(f"Resetting workspace from immutable seeds: {env_assets_dir} + {_CONTROLLER_SURFACE} → {workspace_dir}")
    reset_workspace_from_immutable_sources(
        env_assets_dir=env_assets_dir,
        workspace_dir=workspace_dir,
        controller_surface_dir=_CONTROLLER_SURFACE,
    )
    write_evolution_controller(
        workspace_dir,
        config.evolution_surface,
        contextual_scoping_defense=config.contextual_scoping_defense,
        compartmentalize_evolution_updates=config.compartmentalize_evolution_updates,
        salient_evolution=config.salient_evolution,
    )
    if resume_run_dir is not None and initial_results and resume_meta.get("same_sequence"):
        restored = _restore_workspace_checkpoint(
            run_dir=resume_run_dir,
            workspace_dir=workspace_dir,
            checkpoint_workspace=str(resume_meta.get("checkpoint_workspace") or ""),
        )
        if restored:
            print(
                "Restored evolved-agent workspace checkpoint from last completed task: "
                f"{resume_meta.get('checkpoint_workspace')}"
            )
        else:
            print(
                "WARNING: could not restore evolved-agent workspace checkpoint for in-sequence resume; "
                "continuing from seeded workspace."
            )

    # Wire up components
    guard = SandboxGuard(workspace_dir)
    tracker = EvolutionTracker(log_dir / "benchmark_events.jsonl")
    registry = ToolRegistry()

    for name, handler, desc, params in make_filesystem_tools(guard, tracker, get_allowed_extension_files(config.evolution_surface)):
        registry.register(name, handler, desc, params)

    for name, handler, desc, params in make_memory_tools(workspace_dir):
        registry.register(name, handler, desc, params)

    for name, handler, desc, params in load_created_tools(workspace_dir):
        registry.register(name, handler, desc, params)

    for name, handler, desc, params in make_self_improve_tools(
        workspace_dir,
        tracker,
        registry,
        evolution_surface=config.evolution_surface,
    ):
        registry.register(name, handler, desc, params)

    registry.register_checkpoint()

    session = Session(config, registry, tracker, workspace_dir)

    for name, handler, desc, params in make_compaction_tool(workspace_dir, session):
        registry.register(name, handler, desc, params)

    session.init()
    restored_session_state = False
    if resume_run_dir is not None and initial_results and resume_meta.get("same_sequence"):
        restored_session_state = _restore_session_state(
            run_dir=resume_run_dir,
            session=session,
            session_state_path=str(resume_meta.get("session_state_path") or ""),
        )
        if restored_session_state:
            print(
                "Restored session checkpoint from last completed task: "
                f"{resume_meta.get('session_state_path')}",
            )
    llm_config = {
        "api_key": config.resolve_api_key(),
        "base_url": config.base_url,
        "agent_model": config.model,
        "judge_model": config.judge_model,
        "model_provider_routing": config.model_provider_routing,
    }
    initialize_validation_dispatcher(task_suite, run_manifest.task_files, workspace_dir, llm_config)
    print(f"Using provider: {config.provider_name} ({config.base_url})")
    print(f"Using agent model: {config.model}")
    print(f"Using judge model: {config.judge_model}")
    print(f"Evolution surface: {config.evolution_surface}")
    print(f"Loaded run config: {run_manifest.path}")
    print(f"Loaded task sequences: {', '.join(path.name for path in run_manifest.task_files)}")

    def _reset_sequence_runtime() -> None:
        print(
            f"Resetting workspace for next sequence: "
            f"{env_assets_dir} + {_CONTROLLER_SURFACE} → {workspace_dir}"
        )
        reset_workspace_from_immutable_sources(
            env_assets_dir=env_assets_dir,
            workspace_dir=workspace_dir,
            controller_surface_dir=_CONTROLLER_SURFACE,
        )
        write_evolution_controller(
            workspace_dir,
            config.evolution_surface,
            contextual_scoping_defense=config.contextual_scoping_defense,
            compartmentalize_evolution_updates=config.compartmentalize_evolution_updates,
            salient_evolution=config.salient_evolution,
        )
        session.reset_conversation()

    loop = RoleAwareAgentLoop(
        session,
        task_suite,
        tracker,
        workspace_dir,
        run_dir,
        _REPO_ROOT,
        base_surface=config.evolution_surface,
        upstream_max_task_retries=config.max_task_retries,
        downstream_max_task_retries=0,
        freeze_self_evolution_after_upstream=True,
        downstream_standalone=args.downstream_standalone,
        reset_runtime_state=_reset_sequence_runtime,
        initial_results=initial_results,
        start_task_index=start_task_index,
        initial_handoff=(
            ""
            if restored_session_state
            else str(resume_meta.get("handoff") or "") if resume_meta.get("same_sequence") else ""
        ),
        resume_sequence_id=str(resume_meta.get("last_sequence_id") or "") if resume_meta.get("same_sequence") else "",
    )
    loop.run()


if __name__ == "__main__":
    main()
