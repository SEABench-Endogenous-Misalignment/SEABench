from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ToolResult:
    success: bool
    output: str  # always a string; JSON-encode structured data before storing here


class CheckpointSignal(Exception):
    """Raised by the checkpoint tool to signal the session loop to stop."""

    def __init__(self, answer: str) -> None:
        self.answer = answer
