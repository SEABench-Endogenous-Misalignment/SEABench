from __future__ import annotations

from typing import Any, Callable

from agent.tools.base import CheckpointSignal, ToolResult


class ToolRegistry:
    """
    Holds all registered tools and generates the tools list for the API.

    Each tool is registered with:
      - a unique function name (what the model calls)
      - a handler callable(**kwargs) -> ToolResult
      - a description string
      - a JSON Schema dict for the parameters
    """

    def __init__(self) -> None:
        self._handlers: dict[str, Callable[..., ToolResult]] = {}
        self._schemas: dict[str, dict[str, Any]] = {}

    def register(
        self,
        name: str,
        handler: Callable[..., ToolResult],
        description: str,
        parameters: dict[str, Any],
    ) -> None:
        self._handlers[name] = handler
        self._schemas[name] = {
            "name": name,
            "description": description,
            "parameters": parameters,
        }

    def register_checkpoint(self) -> None:
        """Register the special checkpoint tool that raises CheckpointSignal."""

        def checkpoint(answer: str) -> ToolResult:
            raise CheckpointSignal(answer)

        self.register(
            name="checkpoint",
            handler=checkpoint,
            description=(
                "Submit your final answer for the current task. "
                "Call this exactly once when you have finished the task. "
                "The task ends immediately and the next task begins."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "Your complete answer to the task.",
                    }
                },
                "required": ["answer"],
            },
        )

    def dispatch(self, name: str, args: dict[str, Any]) -> ToolResult:
        """
        Call a registered tool by name with the given arguments.
        Propagates CheckpointSignal — callers must handle it.
        """
        if name not in self._handlers:
            return ToolResult(success=False, output=f"error: unknown tool '{name}'")
        try:
            return self._handlers[name](**args)
        except CheckpointSignal:
            raise  # let the session loop catch this
        except TypeError as exc:
            return ToolResult(success=False, output=f"error: bad arguments — {exc}")
        except Exception as exc:
            return ToolResult(success=False, output=f"error: {type(exc).__name__}: {exc}")

    def tools_for_api(self) -> list[dict[str, Any]]:
        """Return the full tools list expected by the OpenAI chat completions API."""
        return [
            {"type": "function", "function": schema}
            for schema in self._schemas.values()
        ]

    def tools_for_api_filtered(
        self,
        *,
        allowlist: list[str] | None = None,
        denylist: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Return tools for the API with optional allow/deny filtering.

        This only affects what the model can call (tool schemas exposed).
        """
        allowed = set(allowlist) if allowlist else None
        denied = set(denylist or [])

        tools: list[dict[str, Any]] = []
        for name, schema in self._schemas.items():
            if name in denied:
                continue
            if allowed is not None and name not in allowed:
                continue
            tools.append({"type": "function", "function": schema})
        return tools
