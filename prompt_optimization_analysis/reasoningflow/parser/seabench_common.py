"""Shared paths and lossless SEABench-to-ReasoningFlow conversion."""
from __future__ import annotations

import json
import re
from pathlib import Path

RF_ROOT = Path(__file__).resolve().parents[1]
PROMPT_ANALYSIS_ROOT = RF_ROOT.parent
SEABENCH_ROOT = PROMPT_ANALYSIS_ROOT.parent
DEFAULT_OUT = PROMPT_ANALYSIS_ROOT / "out" / "reasoningflow_original"
DEFAULT_TRACES = DEFAULT_OUT / "traces"


def iter_trace_files(root: Path):
    yield from sorted(root.glob("*/*/*/*.json"))


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def document_id(relative_path: Path) -> str:
    return "__".join(relative_path.with_suffix("").parts)


def trace_has_reasoning(trace: dict) -> bool:
    """Whether any turn carries non-empty generated reasoning text."""
    return any(
        str(item.get("reasoning") or "").strip()
        for item in trace.get("turns", [])
    )


def convert_trace(trace: dict, relative_path: Path) -> dict:
    """Convert extracted trace without exposing its answer or benchmark outcome."""
    blocks = []
    spans = []
    cursor = 0
    for item in trace.get("turns", []):
        text = str(item.get("reasoning") or "").strip()
        if not text:
            continue
        if blocks:
            cursor += 2
        start = cursor
        blocks.append(text)
        cursor += len(text)
        spans.append({"turn": item.get("turn"), "start": start, "end": cursor})
    response = "\n\n".join(blocks)
    meta = dict(trace.get("meta") or {})
    # These fields are needed for routing, but no observed outcome is copied.
    public_meta = {
        key: meta.get(key) for key in (
            "model", "surface", "category", "harmtype", "arm", "task_id", "task_dir",
            "selected_candidate", "attempt", "n_attempts",
        ) if key in meta
    }
    public_meta.update({
        "source_trace": relative_path.as_posix(),
        "turn_spans": spans,
        "generator": meta.get("model", "unknown"),
        "domain": f"{meta.get('category', 'unknown')}_{meta.get('harmtype', 'unknown')}",
        "annotator": None,
        "is_human_annotated": False,
    })
    return {
        "doc_id": document_id(relative_path),
        "raw_text": {
            "question": str(trace.get("task_prompt") or ""),
            "response": response,
        },
        "metadata": public_meta,
        "nodes": [],
        "edges": [],
    }


def numeric_id(node_id: str) -> tuple[int, int]:
    match = re.fullmatch(r"(ctx|resp)(\d+)", node_id)
    if not match:
        raise ValueError(f"invalid node id {node_id!r}")
    return (0 if match.group(1) == "ctx" else 1, int(match.group(2)))
