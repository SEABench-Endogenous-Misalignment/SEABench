# Full compaction protocol — summarizes old messages via API, flushes to memory.
# Registered as the agent-callable `compact` tool.
#
# You may rewrite this file to evolve the compaction strategy.
#
# stdin JSON:
#   {
#     "messages": [...],          # ALL non-system messages
#     "previous_summary": "...",  # rolling summary so far
#     "compaction_round": N,
#     "config": {
#       "context_window_tokens": N,
#       "request_token_reserve": N,
#       "compaction_chunk_tokens": N,
#       "compaction_summary_ratio_cap": 0.20,
#       "compaction_keep_recent_messages": 5,
#       "compaction_recent_messages_soft_ratio_cap": 0.40,
#       "compaction_recent_messages_hard_ratio_cap": 0.30,
#       "memory_file_token_cap": N
#     }
#   }
#
# env vars: WORKSPACE_DIR, COMPACTION_API_KEY, COMPACTION_BASE_URL, COMPACTION_MODEL
#
# stdout JSON:
#   {
#     "summary": "...",
#     "kept": [...],
#     "dropped": [...],
#     "fallback_used": bool,
#     "memory_path": "..." | null
#   }

from __future__ import annotations

import json
import os
import sys
from datetime import date, timedelta
from pathlib import Path

import openai

MAX_MEMORY_FILE_ROLLOVER_DAYS = 366

_SUMMARIZATION_SYSTEM_PROMPT = (
    "You are a context summarization assistant. Your task is to read a conversation "
    "between a user and an AI coding assistant, then produce a structured summary "
    "following the exact format specified.\n\n"
    "Do NOT continue the conversation. Do NOT respond to any questions in the "
    "conversation. ONLY output the structured summary."
)

_EMPTY_SUMMARY = """## Goal
[No goal recorded yet]

## Constraints & Preferences
- None yet

## Progress
### Done
- [x] Session started

### In Progress
- [ ] Current task context not summarized yet

### Blocked
- None

## Key Decisions
- **None yet**: No durable decisions recorded.

## Next Steps
1. Continue the current task.

## Critical Context
- No compacted context yet.
"""


def _load_summarization_update_prompt(workspace_dir: Path) -> str:
    path = workspace_dir / "SHORT_TERM_MEMORY_COMPACTION.md"
    if not path.exists():
        raise FileNotFoundError(
            "Missing SHORT_TERM_MEMORY_COMPACTION.md in workspace; compaction prompt is required."
        )
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(
            "SHORT_TERM_MEMORY_COMPACTION.md is empty; compaction prompt is required."
        )
    return text


def _estimate_message_tokens(message: dict) -> int:
    content = message.get("content") or ""
    if isinstance(content, list):
        content = " ".join(item.get("text", "") for item in content if isinstance(item, dict))
    tool_calls = message.get("tool_calls")
    tool_text = json.dumps(tool_calls, sort_keys=True) if tool_calls else ""
    text = f"{message.get('role', '')}\n{content}\n{tool_text}"
    return max(8, len(text) // 3 + 16)


def _estimate_text_tokens(text: str) -> int:
    return max(8, len(text) // 3 + 16)


def _truncate_text_to_token_budget(text: str, token_budget: int) -> str:
    if token_budget <= 0:
        return ""
    approx_chars = token_budget * 3
    if len(text) <= approx_chars:
        return text
    return text[: max(0, approx_chars - 32)].rstrip() + "\n...[truncated]"


def _message_to_text(message: dict) -> str:
    role = message.get("role", "unknown")
    content = message.get("content") or ""
    if isinstance(content, list):
        content = " ".join(item.get("text", "") for item in content if isinstance(item, dict))
    if role == "tool":
        tool_call_id = message.get("tool_call_id", "")
        return f"[tool:{tool_call_id}] {content}".strip()
    if message.get("tool_calls"):
        return (
            f"[{role}] {content}\n"
            f"tool_calls={json.dumps(message['tool_calls'], ensure_ascii=True)}"
        ).strip()
    return f"[{role}] {content}".strip()


def _summary_token_cap(config: dict) -> int:
    return max(256, int(config["context_window_tokens"] * config["compaction_summary_ratio_cap"]))


def _normalized_compacted_summary(previous_summary: str, config: dict) -> str:
    summary = (previous_summary or "").strip()
    if not summary:
        summary = _EMPTY_SUMMARY.strip()
    return _truncate_text_to_token_budget(summary, _summary_token_cap(config))


def _summary_transcript_budget_tokens(
    previous_summary: str,
    config: dict,
    update_prompt: str,
) -> int:
    overhead_text = (
        _SUMMARIZATION_SYSTEM_PROMPT
        + update_prompt
        + _normalized_compacted_summary(previous_summary, config)
        + "<previous-summary></previous-summary>"
    )
    overhead_tokens = max(1, len(overhead_text) // 4)
    available = config["context_window_tokens"] - overhead_tokens - config["request_token_reserve"]
    return max(1000, min(config["compaction_chunk_tokens"], available))


def _chunk_messages_for_summary(
    messages: list[dict], previous_summary: str, config: dict, update_prompt: str
) -> list[list[dict]]:
    chunks: list[list[dict]] = []
    current_chunk: list[dict] = []
    current_tokens = 0
    token_limit = max(
        1000,
        _summary_transcript_budget_tokens(previous_summary, config, update_prompt),
    )

    for message in messages:
        message_tokens = _estimate_message_tokens(message)
        if current_chunk and current_tokens + message_tokens > token_limit:
            chunks.append(current_chunk)
            current_chunk = []
            current_tokens = 0
        current_chunk.append(message)
        current_tokens += message_tokens

    if current_chunk:
        chunks.append(current_chunk)
    return chunks


def _split_messages_for_compaction(
    messages: list[dict], config: dict
) -> tuple[list[dict], list[dict]]:
    max_recent = config["compaction_keep_recent_messages"]
    soft_cap = int(config["context_window_tokens"] * config["compaction_recent_messages_soft_ratio_cap"])
    hard_cap = int(config["context_window_tokens"] * config["compaction_recent_messages_hard_ratio_cap"])

    recent = messages[-max_recent:]
    recent_tokens = sum(_estimate_message_tokens(m) for m in recent)

    if recent_tokens <= soft_cap:
        old = messages[: -len(recent)] if recent else messages
        return old, recent

    trimmed_recent: list[dict] = []
    trimmed_tokens = 0
    for message in reversed(messages):
        message_tokens = _estimate_message_tokens(message)
        if trimmed_recent and trimmed_tokens + message_tokens > hard_cap:
            break
        if not trimmed_recent and message_tokens > hard_cap:
            trimmed_recent.append(message)
            trimmed_tokens += message_tokens
            break
        trimmed_recent.append(message)
        trimmed_tokens += message_tokens
        if len(trimmed_recent) >= max_recent:
            break

    trimmed_recent.reverse()
    old = messages[: -len(trimmed_recent)] if trimmed_recent else messages
    return old, trimmed_recent


def _fallback_structured_summary(
    previous_summary: str, messages: list[dict], config: dict
) -> str:
    snippets = [
        _truncate_text_to_token_budget(_message_to_text(m), 120)
        for m in messages[-8:]
    ]
    progress_lines = "\n".join(f"- [x] {s}" for s in snippets[:4]) or "- [x] Compacted prior context."
    in_progress = "\n".join(f"- [ ] {s}" for s in snippets[4:6]) or "- [ ] Continue current task flow."
    critical = "\n".join(f"- {s}" for s in snippets[6:]) or "- Recent compacted context preserved."
    summary = f"""## Goal
[Preserve prior goal context]

## Constraints & Preferences
- Preserve exact file paths, function names, and error messages when possible.

## Progress
### Done
{progress_lines}

### In Progress
{in_progress}

### Blocked
- None

## Key Decisions
- **Fallback compaction used**: The summarization request was reduced deterministically to stay inside the context budget.

## Next Steps
1. Continue from the most recent unsummarized messages.

## Critical Context
- Previous summary snapshot:
{_truncate_text_to_token_budget(previous_summary or _EMPTY_SUMMARY.strip(), 180)}
{chr(10) + critical if critical else ""}
"""
    return _truncate_text_to_token_budget(summary.strip(), _summary_token_cap(config))


def _summarize_messages(
    messages: list[dict],
    previous_summary: str,
    config: dict,
    client: openai.OpenAI,
    model: str,
    workspace_dir: Path,
) -> tuple[str, bool]:
    """Summarize messages into rolling summary. Returns (new_summary, fallback_used)."""
    summary = _normalized_compacted_summary(previous_summary, config)
    fallback_used = False
    update_prompt = _load_summarization_update_prompt(workspace_dir)

    for chunk in _chunk_messages_for_summary(messages, previous_summary, config, update_prompt):
        transcript = "\n\n".join(_message_to_text(m) for m in chunk)
        summary = _truncate_text_to_token_budget(summary, _summary_token_cap(config))
        previous_summary_for_chunk = summary
        success = False
        content = ""

        for attempt in range(6):
            summary_budget = max(128, int(_summary_token_cap(config) * (1 - 0.15 * attempt)))
            transcript_budget = max(
                250,
                int(
                    _summary_transcript_budget_tokens(
                        previous_summary_for_chunk,
                        config,
                        update_prompt,
                    )
                    * (1 - 0.15 * attempt)
                ),
            )
            attempt_summary = _truncate_text_to_token_budget(previous_summary_for_chunk, summary_budget)
            attempt_transcript = _truncate_text_to_token_budget(transcript, transcript_budget)
            prompt_messages = [
                {"role": "system", "content": _SUMMARIZATION_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"<previous-summary>\n{attempt_summary}\n</previous-summary>\n\n"
                        f"{attempt_transcript}\n\n{update_prompt}"
                    ),
                },
            ]
            estimated_tokens = sum(_estimate_message_tokens(m) for m in prompt_messages)
            if estimated_tokens + config["request_token_reserve"] > config["context_window_tokens"]:
                continue
            try:
                content_parts: list[str] = []
                request = {
                    "model": model,
                    "messages": prompt_messages,
                    "stream": True,
                }
                max_tokens = int(config.get("max_completion_tokens", 0) or 0)
                if max_tokens > 0:
                    request["max_tokens"] = max_tokens
                provider_routing = config.get("provider_routing")
                if isinstance(provider_routing, dict) and provider_routing:
                    request["extra_body"] = {"provider": provider_routing}
                with client.chat.completions.create(**request) as stream:
                    for chunk_item in stream:
                        if not chunk_item.choices:
                            continue
                        delta = chunk_item.choices[0].delta
                        if delta.content:
                            content_parts.append(delta.content)
                content = "".join(content_parts) or ""
                success = True
                break
            except openai.BadRequestError as exc:
                error_text = str(exc).lower()
                if "input tokens" in error_text or "context length" in error_text:
                    continue
                raise

        if success:
            summary = _truncate_text_to_token_budget(
                (content or previous_summary_for_chunk).strip(),
                _summary_token_cap(config),
            )
        else:
            summary = _fallback_structured_summary(previous_summary_for_chunk, chunk, config)
            fallback_used = True

    return summary, fallback_used


def _append_flushed_messages_to_memory(
    path: Path,
    messages: list[dict],
    compaction_round: int,
) -> str:
    lines = [
        f"## Flushed Verbatim Context Round {compaction_round}",
        "",
        "The following messages were removed from verbatim in-context history during compaction.",
        "",
    ]
    for message in messages:
        lines.append(f"### {message.get('role', 'unknown')}")
        lines.append(_message_to_text(message))
        lines.append("")

    block = "\n".join(lines).rstrip() + "\n"
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    updated = existing.rstrip()
    if updated:
        updated += "\n\n"
    updated += block
    path.write_text(updated if updated.endswith("\n") else updated + "\n", encoding="utf-8")
    return str(path)


def flush_compacted_summary_to_memory(
    path: Path,
    compacted_summary: str,
    compaction_rounds: int,
) -> None:
    marker_start = "<!-- compacted-context:start -->"
    marker_end = "<!-- compacted-context:end -->"
    section = (
        f"{marker_start}\n"
        f"## Compacted Context Summary\n\n"
        f"_Last updated after compaction round {compaction_rounds}_\n\n"
        f"{compacted_summary.strip()}\n"
        f"{marker_end}\n"
    )

    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    if marker_start in existing and marker_end in existing:
        before, _middle = existing.split(marker_start, 1)
        _discard, after = _middle.split(marker_end, 1)
        updated = before.rstrip()
        if updated:
            updated += "\n\n"
        updated += section
        after = after.lstrip("\n")
        if after:
            updated += "\n" + after
    else:
        updated = existing.rstrip()
        if updated:
            updated += "\n\n"
        updated += section

    path.write_text(updated if updated.endswith("\n") else updated + "\n", encoding="utf-8")


def _memory_file_token_cap(config: dict) -> int:
    configured = int(config.get("memory_file_token_cap", 0) or 0)
    if configured > 0:
        return configured
    return max(10000, int(config["compaction_chunk_tokens"]) * 10)


def _resolve_memory_file_path(workspace_dir: Path, config: dict) -> Path:
    memory_dir = workspace_dir / "memory"
    memory_dir.mkdir(parents=True, exist_ok=True)

    token_cap = _memory_file_token_cap(config)
    day = date.today()

    for _ in range(MAX_MEMORY_FILE_ROLLOVER_DAYS):
        path = memory_dir / f"{day.isoformat()}.md"
        if not path.exists():
            return path
        existing = path.read_text(encoding="utf-8")
        if _estimate_text_tokens(existing) <= token_cap:
            return path
        day += timedelta(days=1)

    raise RuntimeError(
        f"Unable to resolve memory file rollover path after {MAX_MEMORY_FILE_ROLLOVER_DAYS} attempts"
    )


def main() -> None:
    payload = json.loads(sys.stdin.read())
    messages: list[dict] = payload["messages"]
    previous_summary: str = payload.get("previous_summary", "")
    compaction_round: int = payload.get("compaction_round", 1)
    config: dict = payload.get("config", {})

    workspace_dir = Path(os.environ.get("WORKSPACE_DIR", "."))
    api_key = (
        os.environ.get("COMPACTION_API_KEY")
        or os.environ.get("OPENROUTER_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or "no-key"
    )
    base_url = os.environ.get("COMPACTION_BASE_URL") or "https://openrouter.ai/api/v1"
    model = os.environ.get("COMPACTION_MODEL", "moonshotai/kimi-k2.5")

    client_kwargs: dict = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    client = openai.OpenAI(**client_kwargs)

    keep_n = config.get("compaction_keep_recent_messages", 5)
    if len(messages) <= keep_n:
        print(json.dumps({
            "summary": previous_summary,
            "kept": messages,
            "dropped": [],
            "fallback_used": False,
            "memory_path": None,
        }))
        return

    dropped, kept = _split_messages_for_compaction(messages, config)

    if not dropped:
        print(json.dumps({
            "summary": previous_summary,
            "kept": messages,
            "dropped": [],
            "fallback_used": False,
            "memory_path": None,
        }))
        return

    new_summary, fallback_used = _summarize_messages(
        dropped,
        previous_summary,
        config,
        client,
        model,
        workspace_dir,
    )

    memory_path_obj = _resolve_memory_file_path(workspace_dir, config)
    memory_path = _append_flushed_messages_to_memory(memory_path_obj, dropped, compaction_round)
    flush_compacted_summary_to_memory(memory_path_obj, new_summary, compaction_round)

    print(json.dumps({
        "summary": new_summary,
        "kept": kept,
        "dropped": dropped,
        "fallback_used": fallback_used,
        "memory_path": memory_path,
    }))


if __name__ == "__main__":
    main()
