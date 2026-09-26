#!/usr/bin/env python3
"""Replay one edited task using prior state from an existing run."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(os.environ.get("REPO_ROOT", Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(_REPO_ROOT))

from agent.loop import AgentLoop
from agent.session import Session
from agent.tools.core.compaction import make_compaction_tool
from agent.tools.core.created_tools import load_created_tools
from agent.tools.core.filesystem import make_filesystem_tools
from agent.tools.core.memory import make_memory_tools
from agent.tools.core.self_improve import make_self_improve_tools
from agent.tools.registry import ToolRegistry
from analysis_tools.self_improvement.tracker import EvolutionTracker
from runtime.config import Config
from runtime.evolution import (
    allowed_extension_files as get_allowed_extension_files,
    validate_evolution_surface,
    write_evolution_controller,
)
from runtime.provider_loader import load_provider_profile
from runtime.task_loader import load_yaml_task_suite_from_files
from runtime.validation_dispatcher import initialize_validation_dispatcher
from sandbox.workspace import SandboxGuard, reset_workspace_from_immutable_sources

_CONTROLLER_SURFACE = _REPO_ROOT / "agent" / "seeded_files"
_PROVIDERS_DIR = _REPO_ROOT / "configs" / "providers"


def _safe_slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", str(value).strip().lower()).strip("-")
    return slug or "item"


def _short_run_label(value: str) -> str:
    slug = _safe_slug(Path(value).name if value else "fresh")
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:8]
    if slug.startswith("replay-"):
        slug = "replay"
    return f"{slug[:24]}-{digest}"


def _resolve_path(value: str) -> Path:
    raw = Path(value)
    return raw if raw.is_absolute() else (_REPO_ROOT / raw).resolve()


def _read_summary(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "checkpoints" / "run_summary.json"
    if not path.exists():
        raise SystemExit(f"ERROR: missing run summary: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"ERROR: malformed run summary: {path}")
    return payload




def _find_task_index(task_suite: list[Any], *, sequence_id: str, task_id: str) -> int:
    matches = [
        idx
        for idx, task in enumerate(task_suite)
        if task.sequence_id == sequence_id and task.task_id == task_id
    ]
    if not matches:
        raise SystemExit(f"ERROR: task not found in YAML: sequence_id={sequence_id} task_id={task_id}")
    if len(matches) > 1:
        raise SystemExit(f"ERROR: task matched more than once: sequence_id={sequence_id} task_id={task_id}")
    return matches[0]


def _prior_result(summary: dict[str, Any], *, sequence_id: str, task_id: str) -> dict[str, Any] | None:
    prior: dict[str, Any] | None = None
    for result in summary.get("results", []) or []:
        if not isinstance(result, dict):
            continue
        if result.get("sequence_id") != sequence_id:
            continue
        if result.get("task_id") == task_id:
            return prior
        prior = result
    return prior


def _last_result(summary: dict[str, Any]) -> dict[str, Any] | None:
    results = summary.get("results", [])
    if not isinstance(results, list) or not results:
        return None
    for item in reversed(results):
        if isinstance(item, dict):
            return item
    return None


def _handoff_from_result(result: dict[str, Any] | None) -> str:
    if not result:
        return ""
    continuation = result.get("continuation_state")
    if isinstance(continuation, dict):
        exact = str(continuation.get("next_task_handoff") or "").strip()
        if exact:
            return exact
    passed = result.get("passed")
    outcome = "passed" if passed is True else ("failed" if passed is False else "unscored")
    return (
        f"Task {result.get('task_id')} ({result.get('title', '')}) {outcome}. "
        f"Validator: {result.get('feedback', '')}"
    )


def _clip(value: Any, limit: int) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "\n[truncated]"


def _read_stm_context(path_arg: str) -> dict[str, Any]:
    if not path_arg.strip():
        return {"schema_version": 1, "kind": "selected_path_stm_context", "entries": []}
    path = _resolve_path(path_arg)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"ERROR: invalid STM context file: {path}")
    entries = payload.get("entries")
    if not isinstance(entries, list):
        raise SystemExit(f"ERROR: STM context file missing entries list: {path}")
    return payload


def _read_session_state(path_arg: str) -> dict[str, Any] | None:
    if not path_arg.strip():
        return None
    path = _resolve_path(path_arg)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"ERROR: invalid session state file: {path}")
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        raise SystemExit(f"ERROR: session state file missing messages: {path}")
    return payload


def _stm_context_handoff(payload: dict[str, Any]) -> str:
    entries = payload.get("entries") if isinstance(payload.get("entries"), list) else []
    if not entries:
        return ""
    lines = [
        "Selected-path short-term context from prior optimization replays.",
        "Use this as in-context STM only; it is not a durable memory or controller update.",
    ]
    for entry in entries[-8:]:
        if not isinstance(entry, dict):
            continue
        lines.append(
            f"\n[{entry.get('task_id', '')}] {entry.get('title', '')} "
            f"passed={entry.get('passed')}"
        )
        if str(entry.get("compacted_summary") or "").strip():
            lines.append("Compacted STM summary:")
            lines.append(_clip(entry.get("compacted_summary"), 1600))
        if entry.get("compaction_events"):
            lines.append("Compaction events:")
            lines.append(_clip(json.dumps(entry.get("compaction_events"), ensure_ascii=False), 1200))
        lines.append("Task prompt:")
        lines.append(_clip(entry.get("task_prompt"), 1400))
        lines.append("Final answer:")
        lines.append(_clip(entry.get("answer"), 1800))
        if str(entry.get("feedback") or "").strip():
            lines.append("Validator feedback:")
            lines.append(_clip(entry.get("feedback"), 1600))
        if str(entry.get("safety_feedback") or "").strip():
            lines.append("Safety feedback:")
            lines.append(_clip(entry.get("safety_feedback"), 1600))
    return "\n".join(lines)


def _result_compaction_events(result: dict[str, Any]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for attempt in result.get("attempts") or []:
        if not isinstance(attempt, dict):
            continue
        for run_key in ["task_run", "reflection_run"]:
            run = attempt.get(run_key)
            if not isinstance(run, dict):
                continue
            for event in run.get("compaction_events") or []:
                if isinstance(event, dict):
                    events.append(event)
    return events


def _append_stm_context(
    payload: dict[str, Any],
    *,
    target_task: Any,
    result: dict[str, Any],
    safety: dict[str, Any],
    session: Session,
) -> dict[str, Any]:
    entries = [entry for entry in payload.get("entries", []) if isinstance(entry, dict)]
    task_prompt = target_task.get("prompt") if isinstance(target_task, dict) else getattr(target_task, "prompt", "")
    entry = {
        "sequence_id": str(result.get("sequence_id") or ""),
        "task_id": str(result.get("task_id") or ""),
        "title": str(result.get("title") or ""),
        "task_prompt": _clip(task_prompt or result.get("task_prompt"), 4000),
        "passed": result.get("passed") if isinstance(result.get("passed"), bool) else None,
        "answer": _clip(result.get("answer"), 4000),
        "feedback": _clip(result.get("feedback"), 4000),
        "safety_enabled": bool(safety.get("enabled")),
        "safety_passed": safety.get("passed") if isinstance(safety.get("passed"), bool) else None,
        "safety_feedback": _clip(safety.get("feedback"), 4000),
        "reflection": _clip(result.get("reflection"), 2500),
        "compacted_summary": _clip(session.compacted_summary, 4000),
        "compaction_events": _result_compaction_events(result),
    }
    return {
        "schema_version": 1,
        "kind": "selected_path_stm_context",
        "entries": entries + [entry],
    }


def _checkpoint_workspace(result: dict[str, Any] | None) -> str:
    attempts = result.get("attempts") if isinstance(result, dict) else None
    if not isinstance(attempts, list) or not attempts:
        return ""
    last_attempt = attempts[-1]
    if not isinstance(last_attempt, dict):
        return ""
    checkpoint = last_attempt.get("evolved_agent_checkpoint")
    if not isinstance(checkpoint, dict):
        return ""
    return str(checkpoint.get("workspace_root") or "")


def _restore_checkpoint(*, source_run_dir: Path, workspace_dir: Path, workspace_root: str) -> bool:
    if not workspace_root:
        return False
    src_root = source_run_dir / workspace_root
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


def _resume_sequence_id(
    *,
    prior_sequence_id: str,
    same_sequence_prior: bool,
    checkpoint_restored: bool,
    fresh_session: bool,
) -> str:
    """Keep sequence setup from rerunning on a restored same-sequence workspace."""
    if not prior_sequence_id:
        return ""
    if not fresh_session:
        return prior_sequence_id
    if same_sequence_prior and checkpoint_restored:
        return prior_sequence_id
    return ""


def _session_state_path(result: dict[str, Any] | None) -> str:
    checkpoint = result.get("session_state_checkpoint") if isinstance(result, dict) else None
    if not isinstance(checkpoint, dict):
        return ""
    return str(checkpoint.get("session_state_path") or "")


def _apply_provider_profile(config: Config, profile_name: str) -> Config:
    path = _PROVIDERS_DIR / f"{profile_name}.yaml"
    if not path.exists():
        config.provider_name = profile_name
        return config
    profile = load_provider_profile(path)
    config.provider_name = str(profile.get("name", profile_name))
    config.base_url = str(profile.get("base_url", config.base_url))
    config.api_key_env = str(profile.get("api_key_env", config.api_key_env))
    config.api_key_value = str(profile.get("api_key", config.api_key_value))
    config.model = str(profile.get("default_model", config.model))
    config.judge_model = str(profile.get("default_model", config.judge_model or config.model))
    raw_model_caps = profile.get("model_max_completion_tokens")
    if isinstance(raw_model_caps, dict):
        config.model_max_completion_tokens = {
            str(model): int(limit)
            for model, limit in raw_model_caps.items()
            if str(model).strip() and int(limit) > 0
        }
    raw_model_routing = profile.get("model_provider_routing")
    if isinstance(raw_model_routing, dict):
        config.model_provider_routing = {
            str(model): dict(routing)
            for model, routing in raw_model_routing.items()
            if str(model).strip() and isinstance(routing, dict)
        }
    for key in [
        "context_window_tokens",
        "compaction_keep_recent_messages",
        "compaction_chunk_tokens",
        "request_token_reserve",
    ]:
        if profile.get(key) is not None:
            setattr(config, key, int(profile[key]))
    for key in [
        "compaction_trigger_ratio",
        "compaction_recent_messages_soft_ratio_cap",
        "compaction_recent_messages_hard_ratio_cap",
        "compaction_summary_ratio_cap",
    ]:
        if profile.get(key) is not None:
            setattr(config, key, float(profile[key]))
    return config


def _apply_judge_provider_profile(config: Config, profile_name: str, judge_model_override: str) -> Config:
    path = _PROVIDERS_DIR / f"{profile_name}.yaml"
    if not path.exists():
        return config
    profile = load_provider_profile(path)
    config.judge_base_url = str(profile.get("base_url", config.judge_base_url))
    config.judge_api_key_env = str(profile.get("api_key_env", config.judge_api_key_env))
    config.judge_api_key_value = str(profile.get("api_key", config.judge_api_key_value))
    config.judge_model = judge_model_override or str(profile.get("default_model", config.judge_model))
    raw_model_routing = profile.get("model_provider_routing")
    if isinstance(raw_model_routing, dict):
        config.judge_model_provider_routing = {
            str(model): dict(routing)
            for model, routing in raw_model_routing.items()
            if str(model).strip() and isinstance(routing, dict)
        }
    return config


def _build_registry(config: Config, workspace_dir: Path, tracker: EvolutionTracker) -> ToolRegistry:
    guard = SandboxGuard(workspace_dir)
    registry = ToolRegistry()
    for name, handler, desc, params in make_filesystem_tools(
        guard,
        tracker,
        get_allowed_extension_files(config.evolution_surface),
    ):
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
    return registry


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay one edited task from a prior SEABench run state.")
    parser.add_argument("--resume-run-dir", default="", help="Existing run directory to replay from.")
    parser.add_argument("--fresh", action="store_true", help="Run from a fresh workspace instead of a prior run state.")
    parser.add_argument("--task-file", required=True, help="Edited task YAML file to load from the current repo.")
    parser.add_argument("--sequence-id", required=True, help="Sequence id containing the task.")
    parser.add_argument("--task-id", required=True, help="Task id to replay.")
    parser.add_argument("--output-dir", default="artifacts/task_replays", help="Directory for replay artifacts.")
    parser.add_argument("--surface", default="", help="Override evolution surface; defaults to the source run surface.")
    parser.add_argument(
        "--contextual-scoping-defense",
        choices=("inherit", "enabled", "disabled"),
        default="inherit",
        help="Override contextual-scoping guidance for this replay.",
    )
    parser.add_argument(
        "--compartmentalize-evolution-updates",
        choices=("inherit", "enabled", "disabled"),
        default="inherit",
        help="Override advisory controller-update compartment labels for this replay.",
    )
    parser.add_argument(
        "--salient-evolution",
        choices=("inherit", "enabled", "disabled"),
        default="inherit",
        help="Override harness-timestamped self-evolved-update logging for this replay.",
    )
    parser.add_argument(
        "--checkpoint-from-run-dir",
        default="",
        help=(
            "Restore the final evolved workspace checkpoint from this run when the target task "
            "has no same-sequence prior. Used for frozen downstream test evaluation."
        ),
    )
    parser.add_argument(
        "--use-last-result-as-prior",
        action="store_true",
        help=(
            "If the source run does not contain the full same-sequence prefix, use its final "
            "result as the immediate prior task. Used for one-task replay chains."
        ),
    )
    parser.add_argument("--provider-profile", default="", help="Override provider profile; defaults to source provider_name.")
    parser.add_argument("--agent-model", default="", help="Override agent model; defaults to source run agent_model.")
    parser.add_argument("--judge-model", default="", help="Override judge model; defaults to source run judge_model.")
    parser.add_argument(
        "--judge-provider-profile",
        default="openrouter",
        help="Provider profile for the safety/utility judge, independent of --provider-profile.",
    )
    parser.add_argument("--max-task-retries", type=int, default=None, help="Override max task retries for replay.")
    parser.add_argument(
        "--initial-stm-context-file",
        default="",
        help="Optimization-only: selected-path STM transcript/compaction context to inject into this replay.",
    )
    parser.add_argument(
        "--export-stm-context-file",
        default="",
        help="Optimization-only: write selected-path STM context after this replay.",
    )
    parser.add_argument(
        "--initial-session-state-file",
        default="",
        help="Restore an exact prior session state before replaying this task.",
    )
    parser.add_argument(
        "--fresh-session",
        action="store_true",
        help=(
            "Start this replay from a fresh conversation even if a same-sequence prior task exists "
            "or an initial session-state file is available. Workspace state is still restored."
        ),
    )
    parser.add_argument(
        "--export-session-state-file",
        default="",
        help="Write the exact session state after this replay.",
    )
    parser.add_argument(
        "--workspace-init-dir",
        default="",
        help=(
            "Pre-initialized workspace directory. When provided, its contents are copied into the "
            "run workspace and env-assets reset + checkpoint restore are both skipped. "
            "Implies --fresh semantics for prior-result lookup."
        ),
    )
    args = parser.parse_args()

    workspace_init_dir = _resolve_path(args.workspace_init_dir) if args.workspace_init_dir.strip() else None
    if workspace_init_dir is None and not args.fresh and not args.resume_run_dir.strip():
        raise SystemExit("ERROR: provide --resume-run-dir, --fresh, or --workspace-init-dir.")
    fresh_effective = args.fresh or workspace_init_dir is not None
    source_run_dir = _resolve_path(args.resume_run_dir) if args.resume_run_dir.strip() else Path("")
    checkpoint_run_dir = _resolve_path(args.checkpoint_from_run_dir) if args.checkpoint_from_run_dir.strip() else None
    task_file = _resolve_path(args.task_file)
    output_root = _resolve_path(args.output_dir)
    summary = {} if fresh_effective else _read_summary(source_run_dir)
    task_suite = load_yaml_task_suite_from_files([task_file])
    target_idx = _find_task_index(task_suite, sequence_id=args.sequence_id, task_id=args.task_id)
    target_task = task_suite[target_idx]
    prior = None if fresh_effective else _prior_result(summary, sequence_id=args.sequence_id, task_id=args.task_id)
    if prior is None and args.use_last_result_as_prior and not fresh_effective:
        prior = _last_result(summary)
    if fresh_effective and target_idx > 0:
        raise SystemExit("ERROR: --fresh / --workspace-init-dir can only run the first selected task in sequence order.")
    if not fresh_effective and target_idx > 0 and prior is None:
        raise SystemExit(
            "ERROR: requested task is not the first task in the edited YAML, but no prior "
            f"completed task was found in the source run for sequence_id={args.sequence_id}."
        )

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    replay_name = (
        f"replay_{_safe_slug(args.sequence_id)}__{_safe_slug(args.task_id)}__"
        f"{_short_run_label(str(source_run_dir) if not fresh_effective else 'fresh')}__{timestamp}"
    )
    run_dir = output_root / replay_name
    workspace_dir = run_dir / "workspace"
    log_dir = run_dir / "logs" / "event_logs"
    for path in [workspace_dir, log_dir, run_dir / "checkpoints", run_dir / "analysis"]:
        path.mkdir(parents=True, exist_ok=True)

    config = Config()
    provider_profile = args.provider_profile or str(summary.get("provider_name") or config.provider_name)
    config = _apply_provider_profile(config, provider_profile)
    config.evolution_surface = validate_evolution_surface(
        args.surface or str(summary.get("evolution_surface") or config.evolution_surface)
    )
    config.model = args.agent_model or str(summary.get("agent_model") or config.model)
    config.judge_model = args.judge_model or str(summary.get("judge_model") or config.judge_model)
    config = _apply_judge_provider_profile(config, args.judge_provider_profile, args.judge_model)
    if args.contextual_scoping_defense != "inherit":
        config.contextual_scoping_defense = args.contextual_scoping_defense == "enabled"
    if args.compartmentalize_evolution_updates != "inherit":
        config.compartmentalize_evolution_updates = args.compartmentalize_evolution_updates == "enabled"
    if args.salient_evolution != "inherit":
        config.salient_evolution = args.salient_evolution == "enabled"
    config.artifacts_dir = str(run_dir)
    config.workspace_dir = str(workspace_dir)
    if args.max_task_retries is not None:
        config.max_task_retries = int(args.max_task_retries)

    checkpoint_root = ""
    restored = False
    restored_session_state = False
    if workspace_init_dir is not None:
        if not workspace_init_dir.exists():
            raise SystemExit(f"ERROR: --workspace-init-dir not found: {workspace_init_dir}")
        if workspace_dir.exists():
            shutil.rmtree(workspace_dir)
        shutil.copytree(workspace_init_dir, workspace_dir)
        write_evolution_controller(
            workspace_dir,
            config.evolution_surface,
            contextual_scoping_defense=config.contextual_scoping_defense,
            compartmentalize_evolution_updates=config.compartmentalize_evolution_updates,
            salient_evolution=config.salient_evolution,
        )
    else:
        env_assets_dir = Path(config.env_assets_dir)
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
        checkpoint_root = _checkpoint_workspace(prior)
        checkpoint_source_run_dir = source_run_dir
        if not checkpoint_root and checkpoint_run_dir is not None:
            checkpoint_root = _checkpoint_workspace(_last_result(_read_summary(checkpoint_run_dir)))
            checkpoint_source_run_dir = checkpoint_run_dir
        if not fresh_effective:
            restored = _restore_checkpoint(
                source_run_dir=checkpoint_source_run_dir,
                workspace_dir=workspace_dir,
                workspace_root=checkpoint_root,
            )
        write_evolution_controller(
            workspace_dir,
            config.evolution_surface,
            contextual_scoping_defense=config.contextual_scoping_defense,
            compartmentalize_evolution_updates=config.compartmentalize_evolution_updates,
            salient_evolution=config.salient_evolution,
        )

    tracker = EvolutionTracker(log_dir / "benchmark_events.jsonl")
    registry = _build_registry(config, workspace_dir, tracker)
    session = Session(config, registry, tracker, workspace_dir)
    for name, handler, desc, params in make_compaction_tool(workspace_dir, session):
        registry.register(name, handler, desc, params)
    session.init()

    session_state_payload = None if args.fresh_session else _read_session_state(args.initial_session_state_file)
    if session_state_payload is None and not fresh_effective and not args.fresh_session:
        session_state_root = _session_state_path(prior)
        if not session_state_root and checkpoint_run_dir is not None:
            session_state_root = _session_state_path(_last_result(_read_summary(checkpoint_run_dir)))
        if session_state_root:
            session_state_source_run_dir = checkpoint_source_run_dir if checkpoint_root else source_run_dir
            session_state_path = session_state_source_run_dir / session_state_root
            if session_state_path.exists():
                session_state_payload = json.loads(session_state_path.read_text(encoding="utf-8"))
    if session_state_payload is not None:
        session.restore_state(session_state_payload)
        restored_session_state = True

    target_tasks = [target_task]
    llm_config = {
        "api_key": config.resolve_judge_api_key(),
        "base_url": config.judge_base_url,
        "judge_model": config.judge_model,
        "model_provider_routing": config.judge_model_provider_routing,
    }
    initialize_validation_dispatcher(target_tasks, [task_file], workspace_dir, llm_config)
    stm_context = {"schema_version": 1, "kind": "selected_path_stm_context", "entries": []} if args.fresh_session else _read_stm_context(args.initial_stm_context_file)
    prior_sequence_id = str(prior.get("sequence_id") or "") if isinstance(prior, dict) else ""
    target_sequence_id = str(getattr(target_task, "sequence_id", "") or "")
    same_sequence_prior = bool(prior_sequence_id and prior_sequence_id == target_sequence_id)
    initial_handoff = "" if args.fresh_session else (_handoff_from_result(prior) if (same_sequence_prior and not restored_session_state) else "")
    stm_handoff = "" if args.fresh_session else _stm_context_handoff(stm_context)
    if stm_handoff and not restored_session_state and same_sequence_prior and not args.fresh_session:
        initial_handoff = f"{stm_handoff}\n\nImmediate prior task outcome:\n{initial_handoff}".strip()
    resume_sequence_id = _resume_sequence_id(
        prior_sequence_id=prior_sequence_id,
        same_sequence_prior=same_sequence_prior,
        checkpoint_restored=restored,
        fresh_session=args.fresh_session,
    )

    def _reset_sequence_runtime() -> None:
        reset_workspace_from_immutable_sources(
            env_assets_dir=Path(config.env_assets_dir),
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

    loop = AgentLoop(
        session,
        target_tasks,
        tracker,
        workspace_dir,
        run_dir,
        _REPO_ROOT,
        reset_runtime_state=_reset_sequence_runtime,
        initial_results=[],
        start_task_index=0,
        initial_handoff=initial_handoff,
        resume_sequence_id=resume_sequence_id,
    )
    loop.run()

    result = loop.results[-1] if loop.results else {}
    attempts = result.get("attempts") or []
    last_attempt = attempts[-1] if attempts else {}
    safety = last_attempt.get("hidden_safety_observer") or result.get("hidden_safety_observer") or {}
    replay_meta = {
        "source_run_dir": str(source_run_dir) if not fresh_effective else "",
        "fresh": fresh_effective,
        "workspace_init_dir": str(workspace_init_dir) if workspace_init_dir is not None else "",
        "task_file": str(task_file),
        "sequence_id": args.sequence_id,
        "task_id": args.task_id,
        "prior_task_id": prior.get("task_id") if isinstance(prior, dict) else None,
        "restored_checkpoint_workspace": checkpoint_root,
        "checkpoint_from_run_dir": str(checkpoint_source_run_dir) if checkpoint_root else "",
        "restored_checkpoint": restored,
        "run_dir": str(run_dir),
        "provider_name": config.provider_name,
        "agent_model": config.model,
        "judge_model": config.judge_model,
        "evolution_surface": config.evolution_surface,
        "contextual_scoping_defense": config.contextual_scoping_defense,
        "compartmentalize_evolution_updates": config.compartmentalize_evolution_updates,
        "salient_evolution": config.salient_evolution,
    }
    (run_dir / "replay_meta.json").write_text(json.dumps(replay_meta, indent=2), encoding="utf-8")
    (run_dir / "task_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (run_dir / "answer.txt").write_text(str(result.get("answer") or ""), encoding="utf-8")
    (run_dir / "validator_feedback.txt").write_text(str(result.get("feedback") or ""), encoding="utf-8")
    (run_dir / "safety_result.json").write_text(json.dumps(safety, indent=2), encoding="utf-8")
    updated_stm_context = _append_stm_context(
        stm_context,
        target_task=target_task,
        result=result,
        safety=safety,
        session=session,
    )
    stm_context_path = run_dir / "stm_context.json"
    stm_context_path.write_text(json.dumps(updated_stm_context, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.export_stm_context_file.strip():
        export_path = _resolve_path(args.export_stm_context_file)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(stm_context_path, export_path)
    session_state_path = run_dir / "session_state.json"
    session_state_path.write_text(json.dumps(session.export_state(), indent=2, ensure_ascii=False), encoding="utf-8")
    if args.export_session_state_file.strip():
        export_path = _resolve_path(args.export_session_state_file)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(session_state_path, export_path)

    print("\nReplay complete")
    print(f"Artifacts: {run_dir}")
    print(f"Prior task: {replay_meta['prior_task_id'] or '(sequence start)'}")
    print(f"Restored checkpoint: {restored} {checkpoint_root}")
    print(f"Restored session state: {restored_session_state}")
    print(f"Task passed: {result.get('passed')}")
    if safety.get("enabled"):
        print(f"Safety passed: {safety.get('passed')}")
        print(f"Safety feedback: {str(safety.get('feedback') or '')[:1200]}")
    else:
        print("Safety passed: (not enabled)")
    print(f"Answer preview: {str(result.get('answer') or '')[:500].replace(chr(10), ' ')}")


if __name__ == "__main__":
    main()
