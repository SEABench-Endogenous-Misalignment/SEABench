from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class EvolutionTracker:
    """
    Appends one JSON object per line to benchmark_events.jsonl.

    Each event has at minimum:
      - "event": event type string
      - "ts":    ISO-8601 UTC timestamp

    Additional keyword arguments are included as metadata.

    Key event types emitted by the tool layer:
      learning_recorded   — record_learning called successfully
      memory_updated      — update_memory called successfully
      skill_created       — create_skill called successfully  (+ skill_name)
      skill_run           — run_skill called                  (+ skill_name, returncode, success)

    Key event types emitted by the agent loop:
      task_started        — new task beginning               (+ task_id, title)
      task_completed      — checkpoint called                 (+ task_id, answer_preview)
      task_validated      — validator result recorded         (+ task_id, passed, feedback)
      tool_called         — every tool dispatch               (+ tool)
      max_turns_reached   — task ended without checkpoint
      run_started         — beginning of full run
      run_finished        — end of full run                   (+ tasks_total, tasks_passed)
    """

    def __init__(self, log_path: Path) -> None:
        self.log_path = log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event: str, **metadata: Any) -> None:
        row: dict[str, Any] = {
            "event": event,
            "ts": datetime.now(UTC).isoformat(),
            **metadata,
        }
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
