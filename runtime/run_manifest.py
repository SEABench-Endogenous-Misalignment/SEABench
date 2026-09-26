from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from runtime.config import Config
from runtime.provider_loader import load_provider_profile


@dataclass
class RunManifest:
    name: str
    path: Path
    evolution_surface: str | None
    provider_profile: str | None
    agent_model: str | None
    judge_model: str | None
    context_window_tokens: int | None
    compaction_trigger_ratio: float | None
    compaction_keep_recent_messages: int | None
    compaction_recent_messages_soft_ratio_cap: float | None
    compaction_recent_messages_hard_ratio_cap: float | None
    compaction_chunk_tokens: int | None
    compaction_summary_ratio_cap: float | None
    request_token_reserve: int | None
    memory_file_token_cap: int | None
    task_files: list[Path]


def load_run_manifest(path: Path, repo_root: Path) -> RunManifest:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Run config must contain a mapping: {path}")

    tasks = payload.get("tasks", {})
    if not isinstance(tasks, dict):
        raise ValueError(f"'tasks' must be a mapping in {path}")

    task_files = tasks.get("files", [])
    if not isinstance(task_files, list) or not task_files:
        raise ValueError(f"'tasks.files' must be a non-empty list in {path}")

    resolved_task_files = [repo_root / str(item) for item in task_files]
    for task_path in resolved_task_files:
        if not task_path.exists():
            raise ValueError(f"Task file listed in {path} does not exist: {task_path}")

    provider = payload.get("provider", {})
    if provider is None:
        provider = {}
    if not isinstance(provider, dict):
        raise ValueError(f"'provider' must be a mapping in {path}")

    return RunManifest(
        name=str(payload.get("name", path.stem)),
        path=path,
        evolution_surface=str(payload["evolution_surface"]) if payload.get("evolution_surface") else None,
        provider_profile=str(provider["profile"]) if provider.get("profile") else None,
        agent_model=(
            str(provider["agent_model"])
            if provider.get("agent_model")
            else (str(provider["model"]) if provider.get("model") else None)
        ),
        judge_model=str(provider["judge_model"]) if provider.get("judge_model") else None,
        context_window_tokens=int(payload["context_window_tokens"]) if payload.get("context_window_tokens") is not None else None,
        compaction_trigger_ratio=float(payload["compaction_trigger_ratio"]) if payload.get("compaction_trigger_ratio") is not None else None,
        compaction_keep_recent_messages=int(payload["compaction_keep_recent_messages"]) if payload.get("compaction_keep_recent_messages") is not None else None,
        compaction_recent_messages_soft_ratio_cap=float(payload["compaction_recent_messages_soft_ratio_cap"]) if payload.get("compaction_recent_messages_soft_ratio_cap") is not None else None,
        compaction_recent_messages_hard_ratio_cap=float(payload["compaction_recent_messages_hard_ratio_cap"]) if payload.get("compaction_recent_messages_hard_ratio_cap") is not None else None,
        compaction_chunk_tokens=int(payload["compaction_chunk_tokens"]) if payload.get("compaction_chunk_tokens") is not None else None,
        compaction_summary_ratio_cap=float(payload["compaction_summary_ratio_cap"]) if payload.get("compaction_summary_ratio_cap") is not None else None,
        request_token_reserve=int(payload["request_token_reserve"]) if payload.get("request_token_reserve") is not None else None,
        memory_file_token_cap=int(payload["memory_file_token_cap"]) if payload.get("memory_file_token_cap") is not None else None,
        task_files=resolved_task_files,
    )


def apply_run_manifest(config: Config, manifest: RunManifest, providers_dir: Path) -> Config:
    profile_name = manifest.provider_profile or config.provider_name
    profile_path = providers_dir / f"{profile_name}.yaml"
    if profile_path.exists():
        profile = load_provider_profile(profile_path)
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
        if profile.get("context_window_tokens") is not None:
            config.context_window_tokens = int(profile["context_window_tokens"])
        if profile.get("compaction_trigger_ratio") is not None:
            config.compaction_trigger_ratio = float(profile["compaction_trigger_ratio"])
        if profile.get("compaction_keep_recent_messages") is not None:
            config.compaction_keep_recent_messages = int(profile["compaction_keep_recent_messages"])
        if profile.get("compaction_recent_messages_soft_ratio_cap") is not None:
            config.compaction_recent_messages_soft_ratio_cap = float(profile["compaction_recent_messages_soft_ratio_cap"])
        if profile.get("compaction_recent_messages_hard_ratio_cap") is not None:
            config.compaction_recent_messages_hard_ratio_cap = float(profile["compaction_recent_messages_hard_ratio_cap"])
        if profile.get("compaction_chunk_tokens") is not None:
            config.compaction_chunk_tokens = int(profile["compaction_chunk_tokens"])
        if profile.get("compaction_summary_ratio_cap") is not None:
            config.compaction_summary_ratio_cap = float(profile["compaction_summary_ratio_cap"])
        if profile.get("request_token_reserve") is not None:
            config.request_token_reserve = int(profile["request_token_reserve"])
        if profile.get("memory_file_token_cap") is not None:
            config.memory_file_token_cap = int(profile["memory_file_token_cap"])
    else:
        config.provider_name = profile_name

    if manifest.agent_model:
        config.model = manifest.agent_model
    if manifest.judge_model:
        config.judge_model = manifest.judge_model
    if manifest.evolution_surface:
        config.evolution_surface = manifest.evolution_surface
    if manifest.context_window_tokens is not None:
        config.context_window_tokens = manifest.context_window_tokens
    if manifest.compaction_trigger_ratio is not None:
        config.compaction_trigger_ratio = manifest.compaction_trigger_ratio
    if manifest.compaction_keep_recent_messages is not None:
        config.compaction_keep_recent_messages = manifest.compaction_keep_recent_messages
    if manifest.compaction_recent_messages_soft_ratio_cap is not None:
        config.compaction_recent_messages_soft_ratio_cap = manifest.compaction_recent_messages_soft_ratio_cap
    if manifest.compaction_recent_messages_hard_ratio_cap is not None:
        config.compaction_recent_messages_hard_ratio_cap = manifest.compaction_recent_messages_hard_ratio_cap
    if manifest.compaction_chunk_tokens is not None:
        config.compaction_chunk_tokens = manifest.compaction_chunk_tokens
    if manifest.compaction_summary_ratio_cap is not None:
        config.compaction_summary_ratio_cap = manifest.compaction_summary_ratio_cap
    if manifest.request_token_reserve is not None:
        config.request_token_reserve = manifest.request_token_reserve
    if manifest.memory_file_token_cap is not None:
        config.memory_file_token_cap = manifest.memory_file_token_cap

    return config
