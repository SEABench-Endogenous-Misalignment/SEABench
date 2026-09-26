from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass, field


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Config:
    # Provider / model
    provider_name: str = field(
        default_factory=lambda: os.environ.get("PROVIDER_NAME", "openrouter")
    )
    model: str = field(
        default_factory=lambda: os.environ.get("MODEL", "moonshotai/kimi-k2.5")
    )
    judge_model: str = field(
        default_factory=lambda: os.environ.get("JUDGE_MODEL", os.environ.get("MODEL", "moonshotai/kimi-k2.5"))
    )
    base_url: str = field(
        default_factory=lambda: os.environ.get(
            "BASE_URL",
            "https://openrouter.ai/api/v1",
        )
    )
    api_key_env: str = field(
        default_factory=lambda: os.environ.get("API_KEY_ENV", "OPENROUTER_API_KEY")
    )
    api_key_value: str = field(
        default_factory=lambda: os.environ.get("API_KEY", "")
    )

    # Judge provider — independent of the agent's provider above, so the
    # safety/utility judge can run on a different backend than the agent-under-test.
    judge_base_url: str = field(
        default_factory=lambda: os.environ.get(
            "BASE_URL",
            "https://openrouter.ai/api/v1",
        )
    )
    judge_api_key_env: str = field(
        default_factory=lambda: os.environ.get("API_KEY_ENV", "OPENROUTER_API_KEY")
    )
    judge_api_key_value: str = field(
        default_factory=lambda: os.environ.get("API_KEY", "")
    )

    # Tuning
    max_turns_per_task: int = 30  # hard cap on tool-call rounds per task
    model_max_completion_tokens: dict[str, int] = field(default_factory=dict)
    model_provider_routing: dict[str, dict[str, object]] = field(default_factory=dict)
    judge_model_provider_routing: dict[str, dict[str, object]] = field(default_factory=dict)
    max_task_retries: int = field(
        default_factory=lambda: int(os.environ.get("MAX_TASK_RETRIES", "2"))
    )
    context_window_tokens: int = field(
        default_factory=lambda: int(os.environ.get("CONTEXT_WINDOW_TOKENS", "32768"))
    )
    compaction_trigger_ratio: float = field(
        default_factory=lambda: float(os.environ.get("COMPACTION_TRIGGER_RATIO", "0.65"))
    )
    compaction_keep_recent_messages: int = field(
        default_factory=lambda: int(os.environ.get("COMPACTION_KEEP_RECENT_MESSAGES", "3"))
    )
    compaction_recent_messages_soft_ratio_cap: float = field(
        default_factory=lambda: float(os.environ.get("COMPACTION_RECENT_MESSAGES_SOFT_RATIO_CAP", "0.40"))
    )
    compaction_recent_messages_hard_ratio_cap: float = field(
        default_factory=lambda: float(os.environ.get("COMPACTION_RECENT_MESSAGES_HARD_RATIO_CAP", "0.30"))
    )
    compaction_chunk_tokens: int = field(
        default_factory=lambda: int(os.environ.get("COMPACTION_CHUNK_TOKENS", "6000"))
    )
    compaction_summary_ratio_cap: float = field(
        default_factory=lambda: float(os.environ.get("COMPACTION_SUMMARY_RATIO_CAP", "0.20"))
    )
    request_token_reserve: int = field(
        default_factory=lambda: int(os.environ.get("REQUEST_TOKEN_RESERVE", "4096"))
    )
    memory_file_token_cap: int = field(
        default_factory=lambda: int(os.environ.get("MEMORY_FILE_TOKEN_CAP", "0"))
    )
    evolution_surface: str = field(
        default_factory=lambda: os.environ.get("EVOLUTION_SURFACE", "controller_update")
    )
    contextual_scoping_defense: bool = field(
        default_factory=lambda: _env_flag("CONTEXTUAL_SCOPING_DEFENSE")
    )
    compartmentalize_evolution_updates: bool = field(
        default_factory=lambda: _env_flag("COMPARTMENTALIZE_EVOLUTION_UPDATES")
    )
    salient_evolution: bool = field(
        default_factory=lambda: _env_flag("SALIENT_EVOLUTION")
    )

    # Paths — set by scripts via environment variables; defaults work for local dev
    workspace_dir: str = field(
        default_factory=lambda: os.environ.get("WORKSPACE_DIR", "")
    )
    artifacts_dir: str = field(
        default_factory=lambda: os.environ.get(
            "ARTIFACTS_DIR",
            str((Path(__file__).resolve().parent.parent / "artifacts")),
        )
    )
    env_assets_dir: str = field(
        default_factory=lambda: os.environ.get(
            "ENV_ASSETS_DIR",
            str((Path(__file__).resolve().parent.parent / "env_assets")),
        )
    )

    def resolve_api_key(self) -> str:
        if self.api_key_value:
            return self.api_key_value
        if self.api_key_env:
            return os.environ.get(self.api_key_env, "")
        return ""

    def resolve_judge_api_key(self) -> str:
        if self.judge_api_key_value:
            return self.judge_api_key_value
        if self.judge_api_key_env:
            return os.environ.get(self.judge_api_key_env, "")
        return ""
