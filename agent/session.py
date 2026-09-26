from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import openai

from agent.prompts import build_system_prompt
from agent.tools.core.compaction import _run_compaction_extension
from analysis_tools.self_improvement.tracker import EvolutionTracker
from agent.tools.base import CheckpointSignal
from agent.tools.registry import ToolRegistry


class Session:
    """
    Manages the single continuous conversation with the model.

    Key OpenClaw-style behaviours:
      - System prompt is rebuilt from controller files on disk before every task,
        so any apply_controller_update the agent made is picked up immediately.
      - When messages exceed the compaction threshold the oldest are summarised
        into a running text block that is re-injected into the system prompt.
      - The same messages list runs across all tasks (single continuous session).
    """

    def __init__(
        self,
        config: Any,
        registry: ToolRegistry,
        tracker: EvolutionTracker,
        workspace_dir: Path,
    ) -> None:
        self.config = config
        self.registry = registry
        self.tracker = tracker
        self.workspace_dir = workspace_dir
        self.messages: list[dict[str, Any]] = []
        self.compacted_summary: str = ""
        self.compaction_rounds: int = 0
        self._total_tokens_used: int = 0  # cumulative across all API calls
        self._last_prompt_tokens: int = 0  # actual prompt tokens from last API call
        self._messages_at_last_call: int = 0  # len(messages) at that call
        self._task_usage: list[dict[str, Any]] = []  # per-task usage records
        self._current_task_id: str = ""
        self._current_task_prompt_tokens: int = 0
        self._current_task_completion_tokens: int = 0
        self._current_task_total_tokens: int = 0
        self._current_task_tool_calls: Counter = Counter()
        self._current_task_tool_call_details: list[dict[str, Any]] = []
        self._current_task_tool_evidence: list[dict[str, Any]] = []
        self._capture_tool_evidence = False
        self._current_task_compaction_events: list[dict[str, Any]] = []
        self._current_task_reasoning_traces: list[dict[str, Any]] = []
        self._current_task_prompt: str = ""
        self._current_task_handoff: str = ""
        self._current_required_tools: list[str] = []
        self._current_required_answer_substrings: list[str] = []
        self._last_task_run_report: dict[str, Any] | None = None
        api_key = config.resolve_api_key()
        if not api_key:
            label = config.api_key_env or "API_KEY"
            raise ValueError(
                f"Missing API key for provider '{config.provider_name}'. "
                f"Set {label} or provide API_KEY."
            )
        self.client = openai.OpenAI(
            base_url=config.base_url,
            api_key=api_key,
        )

    def init(self) -> None:
        """
        Set the initial system prompt from controller files.
        Called once before the first task.
        """
        system_prompt = build_system_prompt(self.workspace_dir, self.compacted_summary)
        self.messages = [{"role": "system", "content": system_prompt}]

    def reset_conversation(self) -> None:
        """
        Start a fresh conversation thread while keeping the same client and
        cumulative usage report structure.
        """
        self.messages = []
        self.compacted_summary = ""
        self.compaction_rounds = 0
        self._total_tokens_used = 0
        self._last_prompt_tokens = 0
        self._messages_at_last_call = 0
        self.init()

    def _rebuild_system_prompt(self, task_query: str = "") -> None:
        """
        Replace the system message in-place with the current controller file state.
        Called at the start of every task so self-edits are visible immediately.
        If task_query is provided, relevant learnings are retrieved and injected.
        """
        new_prompt = build_system_prompt(self.workspace_dir, self.compacted_summary, task_query)
        if self.messages and self.messages[0]["role"] == "system":
            self.messages[0]["content"] = new_prompt
        else:
            self.messages.insert(0, {"role": "system", "content": new_prompt})

    def _estimate_message_tokens(self, message: dict[str, Any]) -> int:
        content = message.get("content") or ""
        if isinstance(content, list):
            content = " ".join(
                item.get("text", "") for item in content if isinstance(item, dict)
            )
        tool_calls = message.get("tool_calls")
        tool_text = json.dumps(tool_calls, sort_keys=True) if tool_calls else ""
        text = f"{message.get('role', '')}\n{content}\n{tool_text}"
        # Deliberately conservative: chat formatting, tool schema, and provider-side
        # wrappers can add noticeable overhead beyond raw character count.
        return max(8, len(text) // 3 + 16)

    def _estimated_context_tokens(self) -> int:
        if self._last_prompt_tokens > 0:
            # Use actual prompt tokens from last API call as baseline, minus tool
            # schema tokens (which _estimated_request_tokens adds back separately).
            baseline = max(0, self._last_prompt_tokens - self._estimate_tools_tokens())
            new_messages = self.messages[self._messages_at_last_call:]
            delta = sum(self._estimate_message_tokens(m) for m in new_messages)
            return baseline + delta
        return sum(self._estimate_message_tokens(message) for message in self.messages)

    def _compaction_config_dict(self) -> dict[str, Any]:
        return {
            "max_completion_tokens": self.config.model_max_completion_tokens.get(self.config.model, 0),
            "provider_routing": self.config.model_provider_routing.get(self.config.model, {}),
            "context_window_tokens": self.config.context_window_tokens,
            "request_token_reserve": self.config.request_token_reserve,
            "compaction_chunk_tokens": self.config.compaction_chunk_tokens,
            "compaction_summary_ratio_cap": self.config.compaction_summary_ratio_cap,
            "compaction_keep_recent_messages": self.config.compaction_keep_recent_messages,
            "compaction_recent_messages_soft_ratio_cap": self.config.compaction_recent_messages_soft_ratio_cap,
            "compaction_recent_messages_hard_ratio_cap": self.config.compaction_recent_messages_hard_ratio_cap,
            "memory_file_token_cap": self.config.memory_file_token_cap,
        }

    def _estimate_tools_tokens(self) -> int:
        schema = json.dumps(
            self.registry.tools_for_api_filtered(
                allowlist=getattr(self, "_current_tool_allowlist", None),
                denylist=getattr(self, "_current_tool_denylist", None),
            ),
            ensure_ascii=True,
            sort_keys=True,
        )
        return max(32, len(schema) // 3 + 64)

    def _estimated_request_tokens(self) -> int:
        return (
            self._estimated_context_tokens()
            + self._estimate_tools_tokens()
            + self.config.request_token_reserve
        )

    def _tool_call_index(self) -> dict[str, dict[str, Any]]:
        index: dict[str, dict[str, Any]] = {}
        for message in self.messages:
            if message.get("role") != "assistant":
                continue
            for tool_call in message.get("tool_calls") or []:
                if not isinstance(tool_call, dict):
                    continue
                tool_call_id = str(tool_call.get("id") or "")
                if not tool_call_id:
                    continue
                fn = tool_call.get("function") or {}
                raw_args = fn.get("arguments") or "{}"
                try:
                    args = json.loads(raw_args) if isinstance(raw_args, str) else {}
                except json.JSONDecodeError:
                    args = {}
                index[tool_call_id] = {
                    "name": str(fn.get("name") or "unknown_tool"),
                    "args": args if isinstance(args, dict) else {},
                }
        return index

    @staticmethod
    def _tool_result_prune_summary(
        *,
        tool_name: str,
        args: dict[str, Any],
        content: str,
    ) -> str:
        detail = ""
        if tool_name == "read_file" and args.get("path"):
            detail = f" path={args.get('path')}"
        elif tool_name in {"write_file"} and args.get("path"):
            detail = f" path={args.get('path')}"
        elif tool_name == "list_files" and args.get("directory"):
            detail = f" directory={args.get('directory')}"
        elif tool_name in {"memory_get", "memory_search"} and args.get("query"):
            detail = f" query={str(args.get('query'))[:120]!r}"
        elif tool_name == "run_skill" and args.get("skill_name"):
            detail = f" skill_name={args.get('skill_name')}"
        elif tool_name == "read_learning" and args.get("filename"):
            detail = f" filename={args.get('filename')}"
        return f"[pruned {tool_name} output:{detail} chars={len(content)}]"

    def _prune_old_tool_results(self) -> None:
        """
        Trim old tool result payloads while preserving tool-call structure.

        OpenClaw-style session pruning is lighter than compaction: the transcript
        shape remains valid, but large old tool outputs stop being resent on every
        model call. Keep the recent tail intact so the model can still consume the
        tool results it just requested in the current task.
        """
        non_system_indices = [
            idx for idx, message in enumerate(self.messages)
            if message.get("role") != "system"
        ]
        keep_recent = max(6, int(self.config.compaction_keep_recent_messages) * 2)
        keep_indices = set(non_system_indices[-keep_recent:])
        tool_call_index = self._tool_call_index()
        pruned_count = 0
        pruned_chars = 0

        for idx, message in enumerate(self.messages):
            if idx in keep_indices or message.get("role") != "tool":
                continue
            content = message.get("content")
            if not isinstance(content, str):
                continue
            if content.startswith("[pruned ") or len(content) < 500:
                continue
            tool_call_id = str(message.get("tool_call_id") or "")
            tool_meta = tool_call_index.get(tool_call_id, {})
            summary = self._tool_result_prune_summary(
                tool_name=str(tool_meta.get("name") or "tool"),
                args=tool_meta.get("args") if isinstance(tool_meta.get("args"), dict) else {},
                content=content,
            )
            message["content"] = summary
            pruned_count += 1
            pruned_chars += len(content) - len(summary)

        if pruned_count:
            self._last_prompt_tokens = 0
            self._messages_at_last_call = 0
            self.tracker.log(
                "tool_results_pruned",
                pruned_messages=pruned_count,
                approx_chars_removed=pruned_chars,
                keep_recent_non_system_messages=keep_recent,
            )
            self._current_task_compaction_events.append({
                "round": self.compaction_rounds,
                "messages_dropped": 0,
                "memory_path": None,
                "reason": "tool_result_pruning",
                "tool_results_pruned": pruned_count,
            })

    def _messages_for_api(self) -> list[dict[str, Any]]:
        """
        Return a provider-safe copy of the transcript.

        Some OpenAI-compatible backends, notably MiniMax behind OpenRouter,
        reject any historical tool result unless it immediately follows the
        assistant tool_call that created the same id. Compaction and emergency
        truncation can split those old blocks, so sanitize only the outbound
        payload while leaving the full local transcript available for logs.
        """
        sanitized: list[dict[str, Any]] = []
        pending_ids: set[str] = set()
        pending_assistant_idx: int | None = None
        dropped_tools = 0
        stripped_assistants = 0

        def close_pending_assistant() -> None:
            nonlocal pending_ids, pending_assistant_idx, stripped_assistants
            if pending_assistant_idx is not None and pending_ids:
                assistant = dict(sanitized[pending_assistant_idx])
                assistant.pop("tool_calls", None)
                if assistant.get("content") is None:
                    assistant["content"] = "[historical tool call omitted from API request]"
                sanitized[pending_assistant_idx] = assistant
                stripped_assistants += 1
            pending_ids = set()
            pending_assistant_idx = None

        for message in self.messages:
            role = message.get("role")
            copied = json.loads(json.dumps(message))
            if copied.get("content") in {None, ""}:
                copied["content"] = "[empty historical message]"
            if role == "assistant":
                close_pending_assistant()
                tool_calls = copied.get("tool_calls") or []
                if tool_calls:
                    pending_ids = {
                        str(tool_call.get("id") or "")
                        for tool_call in tool_calls
                        if isinstance(tool_call, dict) and tool_call.get("id")
                    }
                    pending_assistant_idx = len(sanitized) if pending_ids else None
                sanitized.append(copied)
                continue

            if role == "tool":
                tool_call_id = str(copied.get("tool_call_id") or "")
                if tool_call_id and tool_call_id in pending_ids:
                    sanitized.append(copied)
                    pending_ids.remove(tool_call_id)
                else:
                    dropped_tools += 1
                continue

            close_pending_assistant()
            sanitized.append(copied)

        close_pending_assistant()
        if dropped_tools or stripped_assistants:
            self.tracker.log(
                "api_tool_history_sanitized",
                dropped_orphan_tool_messages=dropped_tools,
                stripped_incomplete_assistant_tool_calls=stripped_assistants,
            )
        return sanitized

    @staticmethod
    def _is_context_overflow_error(exc: Exception) -> bool:
        error_lower = str(exc).lower()
        return (
            "input tokens" in error_lower
            or "input_tokens" in error_lower
            or "context length" in error_lower
            or "maximum input length" in error_lower
            or "please reduce the length of the input prompt" in error_lower
        )

    def _context_fit_limit_tokens(self) -> int:
        """
        Keep a buffer below the provider's hard context limit.

        OpenAI-compatible backends can differ slightly from our estimator because
        they add chat/tool framing internally. Staying under the exact advertised
        window is not enough; a one-token overshoot should trigger compaction
        before the API request, not crash the run.
        """
        margin = max(512, int(self.config.request_token_reserve))
        return max(1024, int(self.config.context_window_tokens) - margin)

    def _extract_response_text(self, response: Any) -> str:
        if isinstance(response, str):
            stripped = response.strip()
            if stripped.startswith("data:"):
                content_parts: list[str] = []
                reasoning_parts: list[str] = []
                for raw_line in stripped.splitlines():
                    line = raw_line.strip()
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if not payload or payload == "[DONE]":
                        continue
                    try:
                        parsed = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    for choice in parsed.get("choices", []) or []:
                        delta = choice.get("delta") or {}
                        if delta.get("content"):
                            content_parts.append(str(delta["content"]))
                        if delta.get("reasoning"):
                            reasoning_parts.append(str(delta["reasoning"]))
                    if parsed.get("content"):
                        content_parts.append(str(parsed["content"]))
                combined = "".join(content_parts).strip()
                if combined:
                    return combined
                combined_reasoning = "".join(reasoning_parts).strip()
                if combined_reasoning:
                    return combined_reasoning
            return response

        if isinstance(response, dict):
            choices = response.get("choices") or []
            if choices:
                message = choices[0].get("message") or {}
                content = message.get("content", "")
                if isinstance(content, list):
                    return " ".join(
                        item.get("text", "") for item in content if isinstance(item, dict)
                    )
                return str(content or "")
            if "content" in response:
                content = response.get("content", "")
                if isinstance(content, list):
                    return " ".join(
                        item.get("text", "") for item in content if isinstance(item, dict)
                    )
                return str(content or "")

        choices = getattr(response, "choices", None)
        if choices:
            first = choices[0]
            message = getattr(first, "message", None)
            if message is not None:
                content = getattr(message, "content", "")
                if isinstance(content, list):
                    return " ".join(
                        item.get("text", "") for item in content if isinstance(item, dict)
                    )
                return str(content or "")

        content = getattr(response, "content", None)
        if isinstance(content, list):
            return " ".join(
                item.get("text", "") for item in content if isinstance(item, dict)
            )
        if content is not None:
            return str(content)

        return ""

    def _write_compaction_snapshot(self) -> str:
        compaction_dir = self.tracker.log_path.parent / "compaction"
        compaction_dir.mkdir(parents=True, exist_ok=True)
        path = compaction_dir / f"round_{self.compaction_rounds:03d}.md"
        path.write_text(
            (
                f"# Compacted Context Round {self.compaction_rounds}\n\n"
                f"{self.compacted_summary.strip()}\n"
            ),
            encoding="utf-8",
        )
        return str(path)

    def _apply_compaction_result(
        self,
        result: dict[str, Any],
        *,
        round_event_name: str,
        round_reason: str | None = None,
        task_query: str = "",
    ) -> bool:
        if result is None or not result.get("dropped"):
            return False

        self.compacted_summary = result["summary"]
        self.compaction_rounds += 1
        kept = result["kept"]
        dropped = result["dropped"]
        compaction_snapshot_path = self._write_compaction_snapshot()

        system = [m for m in self.messages if m["role"] == "system"]
        self.messages = system[:1] + kept
        self._rebuild_system_prompt(task_query=task_query)
        self._last_prompt_tokens = 0
        self._messages_at_last_call = 0
        self._total_tokens_used = self._estimated_request_tokens()

        memory_path = result.get("memory_path")
        event_payload: dict[str, Any] = {
            "round": self.compaction_rounds,
            "messages_dropped": len(dropped),
            "messages_kept": len(kept),
            "recent_tokens": sum(self._estimate_message_tokens(m) for m in kept),
            "fallback_used": result.get("fallback_used", False),
            "memory_path": memory_path,
            "compaction_snapshot": compaction_snapshot_path,
        }
        if round_reason is not None:
            event_payload["reason"] = round_reason
        self.tracker.log(round_event_name, **event_payload)

        self._current_task_compaction_events.append({
            "round": self.compaction_rounds,
            "messages_dropped": len(dropped),
            "memory_path": memory_path,
            "reason": round_reason or round_event_name,
        })
        memory_note = f" Memory written to: {memory_path}." if memory_path else ""
        reason_note = f" ({round_reason})" if round_reason else ""
        self.messages.append({
            "role": "system",
            "content": (
                f"[CONTEXT MANAGER] Compaction round {self.compaction_rounds}{reason_note}: "
                f"{len(dropped)} older messages were dropped to stay within the context window.{memory_note} "
                "Your conversation history has been summarised above. "
                "Some earlier context may no longer be directly visible."
            ),
        })
        return True

    def _emergency_drop_oldest_non_system(
        self,
        *,
        task_query: str,
        reason: str,
        min_drop_messages: int = 1,
    ) -> bool:
        non_system = [m for m in self.messages if m["role"] != "system"]
        if len(non_system) <= 1:
            return False

        hard_cap_tokens = int(
            self.config.context_window_tokens
            * self.config.compaction_recent_messages_hard_ratio_cap
        )
        kept: list[dict[str, Any]] = []
        kept_tokens = 0
        for message in reversed(non_system):
            message_tokens = self._estimate_message_tokens(message)
            if kept and kept_tokens + message_tokens > hard_cap_tokens:
                break
            kept.append(message)
            kept_tokens += message_tokens
        kept.reverse()
        if not kept:
            kept = non_system[-1:]
        max_kept_for_drop = max(1, len(non_system) - max(1, min_drop_messages))
        if len(kept) > max_kept_for_drop:
            kept = non_system[-max_kept_for_drop:]
        dropped = non_system[: len(non_system) - len(kept)]
        if not dropped:
            return False

        truncated_lines = [
            "## Goal",
            "[Emergency truncation used to keep the run alive]",
            "",
            "## Constraints & Preferences",
            "- Preserve the most recent task-relevant context verbatim.",
            "",
            "## Progress",
            "### Done",
            "- [x] Older context was compacted or truncated due to context pressure.",
            "",
            "### In Progress",
            "- [ ] Continue from the latest kept interaction window.",
            "",
            "### Blocked",
            "- None",
            "",
            "## Key Decisions",
            f"- **Emergency truncation**: {reason}",
            "",
            "## Next Steps",
            "1. Continue using the latest retained messages and available workspace state.",
            "",
            "## Critical Context",
            "- Some older messages were dropped without full model-generated summarization to avoid a hard context-window failure.",
        ]
        self.compacted_summary = "\n".join(truncated_lines)
        return self._apply_compaction_result(
            {
                "summary": self.compacted_summary,
                "kept": kept,
                "dropped": dropped,
                "fallback_used": True,
                "memory_path": None,
            },
            round_event_name="compaction_round_emergency",
            round_reason=reason,
            task_query=task_query,
        )

    def _trim_latest_handoff_for_overflow(self, *, reason: str, keep_chars: int = 1200) -> bool:
        for message in reversed(self.messages):
            if message.get("role") != "user":
                continue
            content = message.get("content")
            if not isinstance(content, str):
                return False
            marker = "\n\nPrior task outcome:\n"
            if marker not in content:
                return False
            task_text, handoff = content.split(marker, 1)
            if len(handoff) <= keep_chars:
                return False
            message["content"] = (
                f"{task_text}{marker}"
                f"{handoff[:keep_chars].rstrip()}\n"
                f"[handoff truncated by context manager: {reason}]"
            )
            self._last_prompt_tokens = 0
            self._messages_at_last_call = 0
            self.tracker.log(
                "context_overflow_handoff_trimmed",
                reason=reason,
                kept_chars=keep_chars,
                original_chars=len(handoff),
            )
            self._current_task_compaction_events.append({
                "round": self.compaction_rounds,
                "messages_dropped": 0,
                "memory_path": None,
                "reason": reason,
                "handoff_trimmed": True,
            })
            return True
        return False

    def _trim_compacted_summary_for_overflow(self, *, task_query: str, reason: str) -> bool:
        summary = self.compacted_summary.strip()
        if len(summary) <= 1200:
            return False

        keep_chars = max(1200, len(summary) // 2)
        self.compacted_summary = (
            "[Earlier compacted summary was shortened by the context manager after a hard context-window overflow.]\n\n"
            + summary[-keep_chars:]
        )
        self._rebuild_system_prompt(task_query=task_query)
        self._last_prompt_tokens = 0
        self._messages_at_last_call = 0
        self.tracker.log(
            "context_overflow_summary_trimmed",
            reason=reason,
            kept_chars=keep_chars,
            original_chars=len(summary),
        )
        self._current_task_compaction_events.append({
            "round": self.compaction_rounds,
            "messages_dropped": 0,
            "memory_path": None,
            "reason": reason,
            "summary_trimmed": True,
        })
        return True

    def _drop_extra_system_messages_for_overflow(self, *, reason: str) -> bool:
        system_indices = [
            idx for idx, message in enumerate(self.messages)
            if message.get("role") == "system"
        ]
        if len(system_indices) <= 1:
            return False

        first_system_idx = system_indices[0]
        self.messages = [
            message
            for idx, message in enumerate(self.messages)
            if message.get("role") != "system" or idx == first_system_idx
        ]
        self._last_prompt_tokens = 0
        self._messages_at_last_call = 0
        self.tracker.log(
            "context_overflow_extra_system_messages_dropped",
            reason=reason,
            dropped_system_messages=len(system_indices) - 1,
        )
        self._current_task_compaction_events.append({
            "round": self.compaction_rounds,
            "messages_dropped": len(system_indices) - 1,
            "memory_path": None,
            "reason": reason,
            "extra_system_messages_dropped": True,
        })
        return True

    def _trim_system_prompt_for_overflow(
        self,
        *,
        reason: str,
        keep_chars: int | None = None,
    ) -> bool:
        if not self.messages or self.messages[0].get("role") != "system":
            return False
        content = self.messages[0].get("content")
        if not isinstance(content, str):
            return False

        marker = "\n\n--- Short-term memory summary ---\n"
        if marker in content:
            prefix, summary = content.split(marker, 1)
            if len(summary) > 1200:
                kept = max(1200, min(len(summary) // 2, keep_chars or 6000))
                self.messages[0]["content"] = (
                    f"{prefix}{marker}"
                    "[Short-term memory summary shortened by context manager after context overflow.]\n\n"
                    f"{summary[-kept:]}"
                )
                self._last_prompt_tokens = 0
                self._messages_at_last_call = 0
                self.tracker.log(
                    "context_overflow_system_summary_trimmed",
                    reason=reason,
                    kept_chars=kept,
                    original_chars=len(summary),
                )
                self._current_task_compaction_events.append({
                    "round": self.compaction_rounds,
                    "messages_dropped": 0,
                    "memory_path": None,
                    "reason": reason,
                    "system_summary_trimmed": True,
                })
                return True

        # Last-resort protection for provider/accounting off-by-some errors:
        # preserve the leading controller instructions and drop tail context that
        # was already too large to submit. This is intentionally conservative and
        # only runs after normal compaction, handoff trimming, summary trimming,
        # and extra-system-message removal have failed.
        current_len = len(content)
        target_chars = keep_chars or int(current_len * 0.85)
        target_chars = max(4000, min(target_chars, current_len - 1000))
        if target_chars >= current_len:
            return False
        self.messages[0]["content"] = (
            content[:target_chars].rstrip()
            + "\n\n[System prompt tail shortened by context manager after hard context-window overflow.]"
        )
        self._last_prompt_tokens = 0
        self._messages_at_last_call = 0
        self.tracker.log(
            "context_overflow_system_prompt_trimmed",
            reason=reason,
            kept_chars=target_chars,
            original_chars=current_len,
        )
        self._current_task_compaction_events.append({
            "round": self.compaction_rounds,
            "messages_dropped": 0,
            "memory_path": None,
            "reason": reason,
            "system_prompt_trimmed": True,
        })
        return True

    def _recover_from_context_overflow(self, *, task_query: str, reason: str) -> bool:
        before_round = self.compaction_rounds
        self._ensure_request_fits_context(task_query=task_query)
        self._compact_history_if_needed()
        if self.compaction_rounds != before_round:
            return True

        non_system = [m for m in self.messages if m["role"] != "system"]
        if len(non_system) > self.config.compaction_keep_recent_messages:
            result = _run_compaction_extension(
                workspace_dir=self.workspace_dir,
                messages=non_system,
                previous_summary=self.compacted_summary,
                compaction_round=self.compaction_rounds + 1,
                config=self._compaction_config_dict(),
                api_key=self.config.resolve_api_key(),
                base_url=self.config.base_url,
                model=self.config.model,
            )
            if self._apply_compaction_result(
                result,
                round_event_name="compaction_round_forced",
                round_reason=reason,
                task_query=task_query,
            ):
                return True

        if self._emergency_drop_oldest_non_system(
            task_query=task_query,
            reason=f"{reason}_emergency_truncation",
            min_drop_messages=2,
        ):
            return True
        if self._trim_latest_handoff_for_overflow(reason=f"{reason}_handoff_trim"):
            return True
        if self._trim_compacted_summary_for_overflow(
            task_query=task_query,
            reason=f"{reason}_summary_trim",
        ):
            return True
        if self._drop_extra_system_messages_for_overflow(reason=f"{reason}_system_trim"):
            self._rebuild_system_prompt(task_query=task_query)
            return True
        if self._trim_system_prompt_for_overflow(reason=f"{reason}_system_prompt_trim"):
            return True
        return False

    def _ensure_request_fits_context(self, *, task_query: str) -> None:
        max_attempts = 8
        for _ in range(max_attempts):
            estimated = self._estimated_request_tokens()
            if estimated < self._context_fit_limit_tokens():
                return

            non_system = [m for m in self.messages if m["role"] != "system"]
            if len(non_system) <= 1:
                return

            result = _run_compaction_extension(
                workspace_dir=self.workspace_dir,
                messages=non_system,
                previous_summary=self.compacted_summary,
                compaction_round=self.compaction_rounds + 1,
                config=self._compaction_config_dict(),
                api_key=self.config.resolve_api_key(),
                base_url=self.config.base_url,
                model=self.config.model,
            )
            if self._apply_compaction_result(
                result,
                round_event_name="compaction_round_preflight",
                round_reason="preflight_context_fit",
                task_query=task_query,
            ):
                continue

            if self._emergency_drop_oldest_non_system(
                task_query=task_query,
                reason="preflight_emergency_truncation",
            ):
                continue
            return

    def _compact_history_if_needed(self) -> None:
        trigger_tokens = int(
            self.config.context_window_tokens * self.config.compaction_trigger_ratio
        )
        if self._estimated_request_tokens() < trigger_tokens:
            return

        non_system = [m for m in self.messages if m["role"] != "system"]
        if len(non_system) <= self.config.compaction_keep_recent_messages:
            return

        result = _run_compaction_extension(
            workspace_dir=self.workspace_dir,
            messages=non_system,
            previous_summary=self.compacted_summary,
            compaction_round=self.compaction_rounds + 1,
            config=self._compaction_config_dict(),
            api_key=self.config.resolve_api_key(),
            base_url=self.config.base_url,
            model=self.config.model,
        )
        self._apply_compaction_result(
            result,
            round_event_name="compaction_round",
            round_reason=f"trigger_ratio_{self.config.compaction_trigger_ratio:.2f}",
            task_query="",
        )

    def _stream_completion(
        self,
        *,
        tool_allowlist: list[str] | None = None,
        tool_denylist: list[str] | None = None,
    ) -> tuple[str | None, list[dict[str, Any]] | None, Any, str | None]:
        """
        Call the API with stream=True (Open WebUI always streams regardless of
        the stream parameter) and collect all chunks into (content, tool_calls, usage,
        reasoning). tool_calls entries are already in the dict format used by the
        messages list. reasoning is the model's chain-of-thought trace, when the
        provider exposes one via a `reasoning` delta field (e.g. Kimi K2.5); None
        for providers that don't stream one.
        """
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_accum: dict[int, dict[str, Any]] = {}
        usage = None

        request: dict[str, Any] = {
            "model": self.config.model,
            "messages": self._messages_for_api(),
            "tools": self.registry.tools_for_api_filtered(
                allowlist=tool_allowlist,
                denylist=tool_denylist,
            ),
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        max_tokens = self.config.model_max_completion_tokens.get(self.config.model, 0)
        if max_tokens > 0:
            request["max_tokens"] = max_tokens
        provider_routing = self.config.model_provider_routing.get(self.config.model)
        if provider_routing:
            request["extra_body"] = {"provider": dict(provider_routing)}

        with self.client.chat.completions.create(**request) as stream:
            for chunk in stream:
                if getattr(chunk, "usage", None) is not None:
                    usage = chunk.usage
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta.content:
                    content_parts.append(delta.content)
                reasoning_delta = getattr(delta, "reasoning", None)
                if reasoning_delta:
                    reasoning_parts.append(str(reasoning_delta))
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        idx = tc.index
                        if idx not in tool_accum:
                            tool_accum[idx] = {"id": "", "name": "", "arguments": ""}
                        if tc.id:
                            tool_accum[idx]["id"] = tc.id
                        if tc.function:
                            if tc.function.name:
                                tool_accum[idx]["name"] += tc.function.name
                            if tc.function.arguments:
                                tool_accum[idx]["arguments"] += tc.function.arguments

        content = "".join(content_parts) or None
        reasoning = "".join(reasoning_parts).strip() or None
        tool_calls = (
            [
                {
                    "id": v["id"],
                    "type": "function",
                    "function": {"name": v["name"], "arguments": v["arguments"]},
                }
                for v in (tool_accum[i] for i in sorted(tool_accum))
            ]
            if tool_accum
            else None
        )
        return content, tool_calls, usage, reasoning

    def _sanitize_tool_calls_for_history(
        self,
        tool_calls: list[dict[str, Any]] | None,
    ) -> list[dict[str, Any]] | None:
        if not tool_calls:
            return tool_calls

        sanitized: list[dict[str, Any]] = []
        for tc in tool_calls:
            tc_copy = json.loads(json.dumps(tc))
            fn = tc_copy.get("function") or {}
            raw_args = fn.get("arguments", "")
            if not isinstance(raw_args, str):
                raw_args = json.dumps(raw_args)
            try:
                parsed = json.loads(raw_args)
            except json.JSONDecodeError:
                parsed = {}
            fn["arguments"] = json.dumps(parsed, ensure_ascii=False)
            tc_copy["function"] = fn
            sanitized.append(tc_copy)
        return sanitized

    def _record_usage(self, usage: Any) -> None:
        """Extract token counts from the API response usage object and accumulate."""
        if usage is None:
            return
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0
        total_tokens = getattr(usage, "total_tokens", 0) or 0
        self._total_tokens_used = total_tokens  # reflects current context size
        if prompt_tokens > 0:
            self._last_prompt_tokens = prompt_tokens
            self._messages_at_last_call = len(self.messages)
        self._current_task_prompt_tokens += prompt_tokens
        self._current_task_completion_tokens += completion_tokens
        self._current_task_total_tokens += total_tokens

    def _start_task_usage(
        self,
        task_id: str,
        *,
        task_prompt: str,
        handoff: str,
        required_tools: set[str] | None,
        required_answer_substrings: list[str] | None,
        tool_allowlist: list[str] | None,
        tool_denylist: list[str] | None,
        capture_tool_evidence: bool,
    ) -> None:
        self._current_task_id = task_id
        self._current_task_prompt = task_prompt
        self._current_task_handoff = handoff
        self._current_required_tools = sorted(required_tools or [])
        self._current_required_answer_substrings = list(required_answer_substrings or [])
        self._current_task_prompt_tokens = 0
        self._current_task_completion_tokens = 0
        self._current_task_total_tokens = 0
        self._current_task_tool_calls = Counter()
        self._current_task_tool_call_details = []
        self._current_task_tool_evidence = []
        self._capture_tool_evidence = capture_tool_evidence
        self._current_task_compaction_events = []
        self._current_task_reasoning_traces = []
        self._last_task_run_report = None
        self._current_tool_allowlist = tool_allowlist
        self._current_tool_denylist = tool_denylist

    def _flush_task_usage(self, *, status: str, answer: str) -> None:
        usage_record = {
            "task_id": self._current_task_id,
            "prompt_tokens": self._current_task_prompt_tokens,
            "completion_tokens": self._current_task_completion_tokens,
            "total_tokens": self._current_task_total_tokens,
            "tool_calls": dict(self._current_task_tool_calls),
        }
        self._task_usage.append(usage_record)
        self._last_task_run_report = {
            "task_id": self._current_task_id,
            "status": status,
            "task_prompt": self._current_task_prompt,
            "handoff": self._current_task_handoff,
            "required_tools": list(self._current_required_tools),
            "required_answer_substrings": list(self._current_required_answer_substrings),
            "tool_allowlist": list(self._current_tool_allowlist) if self._current_tool_allowlist is not None else None,
            "tool_denylist": list(self._current_tool_denylist) if self._current_tool_denylist is not None else None,
            "answer": answer,
            "usage": usage_record,
            "tool_calls_made": list(self._current_task_tool_call_details),
            "compaction_events": list(self._current_task_compaction_events),
            "reasoning_traces": list(self._current_task_reasoning_traces),
        }

    def last_task_run_report(self) -> dict[str, Any] | None:
        if self._last_task_run_report is None:
            return None
        return json.loads(json.dumps(self._last_task_run_report))

    def current_task_evidence(self) -> list[dict[str, Any]]:
        """Return complete tool results for the current task's safety judge."""
        return json.loads(json.dumps(self._current_task_tool_evidence))

    @staticmethod
    def _restorable_config_keys() -> list[str]:
        return [
            "max_turns_per_task",
            "context_window_tokens",
            "compaction_trigger_ratio",
            "compaction_keep_recent_messages",
            "compaction_recent_messages_soft_ratio_cap",
            "compaction_recent_messages_hard_ratio_cap",
            "compaction_chunk_tokens",
            "compaction_summary_ratio_cap",
            "request_token_reserve",
        ]

    def export_state(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "messages": json.loads(json.dumps(self.messages)),
            "compacted_summary": self.compacted_summary,
            "compaction_rounds": self.compaction_rounds,
            "last_prompt_tokens": self._last_prompt_tokens,
            "messages_at_last_call": self._messages_at_last_call,
            "total_tokens_used": self._total_tokens_used,
            "task_usage": json.loads(json.dumps(self._task_usage)),
            "config_state": {
                key: getattr(self.config, key)
                for key in self._restorable_config_keys()
            },
        }

    def restore_state(self, payload: dict[str, Any]) -> None:
        if not isinstance(payload, dict):
            raise ValueError("session state payload must be a dict")
        messages = payload.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError("session state payload missing messages")
        if not isinstance(messages[0], dict) or messages[0].get("role") != "system":
            raise ValueError("session state payload must start with a system message")
        config_state = payload.get("config_state")
        if isinstance(config_state, dict):
            for key in self._restorable_config_keys():
                if key in config_state:
                    setattr(self.config, key, config_state[key])
        self.messages = json.loads(json.dumps(messages))
        self.compacted_summary = str(payload.get("compacted_summary") or "")
        self.compaction_rounds = int(payload.get("compaction_rounds", 0) or 0)
        self._last_prompt_tokens = int(payload.get("last_prompt_tokens", 0) or 0)
        self._messages_at_last_call = int(payload.get("messages_at_last_call", 0) or 0)
        self._total_tokens_used = int(payload.get("total_tokens_used", 0) or 0)
        task_usage = payload.get("task_usage")
        self._task_usage = json.loads(json.dumps(task_usage)) if isinstance(task_usage, list) else []

    def usage_data(self) -> dict[str, Any]:
        """Return structured usage data suitable for JSON serialization."""
        grand_prompt = sum(r["prompt_tokens"] for r in self._task_usage)
        grand_completion = sum(r["completion_tokens"] for r in self._task_usage)
        grand_total = sum(r["total_tokens"] for r in self._task_usage)
        grand_tools: Counter = Counter()
        for r in self._task_usage:
            grand_tools.update(r["tool_calls"])
        return {
            "summary": {
                "prompt_tokens": grand_prompt,
                "completion_tokens": grand_completion,
                "total_tokens": grand_total,
                "tool_calls": dict(grand_tools),
            },
            "tasks": list(self._task_usage),
        }

    def usage_report(self) -> str:
        """Return a formatted per-task token and tool-call report."""
        if not self._task_usage:
            return "No usage data recorded."

        # Collect all tool names seen across tasks for the tool call section
        all_tools: list[str] = sorted(
            {t for r in self._task_usage for t in r["tool_calls"]}
        )

        # --- Token table ---
        lines = [
            f"{'Task':<20} {'Prompt':>10} {'Completion':>12} {'Total':>10}",
            "-" * 56,
        ]
        grand_prompt = grand_completion = grand_total = 0
        for r in self._task_usage:
            lines.append(
                f"{r['task_id']:<20} {r['prompt_tokens']:>10,} "
                f"{r['completion_tokens']:>12,} {r['total_tokens']:>10,}"
            )
            grand_prompt += r["prompt_tokens"]
            grand_completion += r["completion_tokens"]
            grand_total += r["total_tokens"]
        lines.append("-" * 56)
        lines.append(
            f"{'TOTAL':<20} {grand_prompt:>10,} {grand_completion:>12,} {grand_total:>10,}"
        )

        # --- Tool call table ---
        if all_tools:
            col_w = max(len(t) for t in all_tools) + 2
            header = f"{'Task':<20}" + "".join(f"{t:>{col_w}}" for t in all_tools) + f"{'TOTAL':>{col_w}}"
            lines += ["", header, "-" * len(header)]
            grand_tool_totals: Counter = Counter()
            for r in self._task_usage:
                calls = r["tool_calls"]
                row_total = sum(calls.values())
                row = f"{r['task_id']:<20}" + "".join(
                    f"{calls.get(t, 0):>{col_w}}" for t in all_tools
                ) + f"{row_total:>{col_w}}"
                lines.append(row)
                grand_tool_totals.update(calls)
            grand_row_total = sum(grand_tool_totals.values())
            lines.append("-" * len(header))
            lines.append(
                f"{'TOTAL':<20}"
                + "".join(f"{grand_tool_totals.get(t, 0):>{col_w}}" for t in all_tools)
                + f"{grand_row_total:>{col_w}}"
            )

        return "\n".join(lines)

    def task_compaction_summary(self) -> str:
        """Return a human-readable summary of compaction events during the current task."""
        if not self._current_task_compaction_events:
            return ""
        total_dropped = sum(e["messages_dropped"] for e in self._current_task_compaction_events)
        rounds = len(self._current_task_compaction_events)
        paths = [e["memory_path"] for e in self._current_task_compaction_events if e.get("memory_path")]
        summary = f"{rounds} compaction round(s), {total_dropped} messages dropped total."
        if paths:
            summary += f" Memory path(s): {', '.join(paths)}."
        return summary

    def send_task(
        self,
        task_prompt: str,
        task_id: str = "",
        handoff: str = "",
        required_tools: set[str] | None = None,
        required_answer_substrings: list[str] | None = None,
        tool_allowlist: list[str] | None = None,
        tool_denylist: list[str] | None = None,
        capture_tool_evidence: bool = False,
    ) -> str:
        """
        Rebuild the system prompt from disk, append the task as a user message,
        and run the tool loop until checkpoint or max_turns.
        handoff: pass/fail summary from the previous task, injected into the prompt.
        required_tools: tool names that must be called successfully before the task
        may complete via checkpoint.
        required_answer_substrings: substrings that must appear in the final
        checkpoint answer before the task may complete.
        Returns the answer string or an error string.
        """
        required_tools = required_tools or set()
        required_answer_substrings = required_answer_substrings or []

        if tool_allowlist is not None:
            # Always allow required tools (reflection enforces record_learning).
            tool_allowlist = sorted(set(tool_allowlist) | set(required_tools))

        self._start_task_usage(
            task_id,
            task_prompt=task_prompt,
            handoff=handoff,
            required_tools=required_tools,
            required_answer_substrings=required_answer_substrings,
            tool_allowlist=tool_allowlist,
            tool_denylist=tool_denylist,
            capture_tool_evidence=capture_tool_evidence,
        )

        # Pick up any controller file changes the agent made during previous tasks.
        # Use the task prompt as the retrieval query for relevant learnings.
        self._rebuild_system_prompt(task_query=task_prompt)

        user_content = task_prompt
        if handoff.strip():
            handoff = handoff.strip()
            max_handoff_chars = 3000
            if len(handoff) > max_handoff_chars:
                handoff = (
                    f"{handoff[:max_handoff_chars].rstrip()}\n"
                    "[handoff truncated before request construction]"
                )
            user_content = f"{task_prompt}\n\nPrior task outcome:\n{handoff.strip()}"

        self.messages.append({"role": "user", "content": user_content})

        for _turn in range(self.config.max_turns_per_task):
            self._prune_old_tool_results()
            self._compact_history_if_needed()
            self._prune_old_tool_results()
            self._ensure_request_fits_context(task_query=task_prompt)

            try:
                content, tool_calls, usage, reasoning = self._stream_completion(
                    tool_allowlist=tool_allowlist,
                    tool_denylist=tool_denylist,
                )
            except openai.BadRequestError as exc:
                if self._is_context_overflow_error(exc):
                    if self._recover_from_context_overflow(
                        task_query=task_prompt,
                        reason="bad_request_context_overflow",
                    ):
                        continue
                raise
            except Exception as exc:
                if self._is_context_overflow_error(exc):
                    if self._recover_from_context_overflow(
                        task_query=task_prompt,
                        reason="request_context_overflow",
                    ):
                        continue
                raise
            self._record_usage(usage)
            if reasoning:
                self._current_task_reasoning_traces.append({
                    "turn": _turn,
                    "reasoning": reasoning,
                })

            assistant_entry: dict[str, Any] = {"role": "assistant", "content": content}
            sanitized_tool_calls = self._sanitize_tool_calls_for_history(tool_calls)
            if sanitized_tool_calls:
                assistant_entry["tool_calls"] = sanitized_tool_calls
            self.messages.append(assistant_entry)

            if not sanitized_tool_calls:
                # Plain text response — nudge the model to call checkpoint
                self.messages.append({
                    "role": "user",
                    "content": "Please call checkpoint(answer=...) with your final answer to complete the task.",
                })
                continue

            tool_results: list[dict[str, Any]] = []
            checkpoint_answer: str | None = None

            for tc in sanitized_tool_calls:
                name = tc["function"]["name"]
                try:
                    args = json.loads(tc["function"]["arguments"])
                except json.JSONDecodeError:
                    args = {}

                self.tracker.log("tool_called", tool=name)
                self._current_task_tool_calls[name] += 1

                try:
                    result = self.registry.dispatch(name, args)
                    tc_content = result.output
                    self._current_task_tool_call_details.append({
                        "tool": name,
                        "args": args,
                        "success": result.success,
                        "output_preview": tc_content[:500],
                        "tool_call_id": tc["id"],
                    })
                    if self._capture_tool_evidence:
                        self._current_task_tool_evidence.append({
                            "tool": name,
                            "args": args,
                            "success": result.success,
                            "output": tc_content,
                            "tool_call_id": tc["id"],
                        })
                except CheckpointSignal as cs:
                    checkpoint_answer = cs.answer
                    tc_content = json.dumps({"status": "answer_received"})
                    self._current_task_tool_call_details.append({
                        "tool": name,
                        "args": args,
                        "success": True,
                        "output_preview": tc_content,
                        "tool_call_id": tc["id"],
                    })

                tool_results.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": tc_content,
                })

            self.messages.extend(tool_results)

            if checkpoint_answer is not None:
                missing_tools = [
                    name for name in sorted(required_tools)
                    if self._current_task_tool_calls.get(name, 0) == 0
                ]
                if missing_tools:
                    missing = ", ".join(missing_tools)
                    self.messages.append({
                        "role": "user",
                        "content": (
                            "Before completing this phase, you must call the following tool(s): "
                            f"{missing}. After doing that, call checkpoint(answer=...) again."
                        ),
                    })
                    continue
                answer_lc = checkpoint_answer.lower()
                missing_substrings = [
                    value for value in required_answer_substrings
                    if value.lower() not in answer_lc
                ]
                if missing_substrings:
                    required_text = ", ".join(missing_substrings)
                    self.messages.append({
                        "role": "user",
                        "content": (
                            "Before completing this phase, your checkpoint answer must include: "
                            f"{required_text}. Update your reflection summary and call "
                            "checkpoint(answer=...) again."
                        ),
                    })
                    continue
                self._flush_task_usage(status="checkpoint", answer=checkpoint_answer)
                self._current_tool_allowlist = None
                self._current_tool_denylist = None
                return checkpoint_answer

        self.tracker.log("max_turns_reached")
        self._flush_task_usage(
            status="max_turns_reached",
            answer="ERROR: max turns reached without checkpoint",
        )
        self._current_tool_allowlist = None
        self._current_tool_denylist = None
        return "ERROR: max turns reached without checkpoint"
