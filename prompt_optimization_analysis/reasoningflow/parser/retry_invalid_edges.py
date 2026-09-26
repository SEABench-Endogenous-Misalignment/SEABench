#!/usr/bin/env python3
"""Retry only destinations whose incoming ReasoningFlow edges violate the schema."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

import llm_labeler as rf
from seabench_common import DEFAULT_OUT, numeric_id, read_json
from validate_seabench import EDGE_RULES


def validate_incoming_edges(doc: dict, destination_id: str, edges: list[dict]) -> list[str]:
    """Validate candidate incoming edges using the same rules as final validation."""
    errors: list[str] = []
    nodes = {node["id"]: node for node in doc.get("nodes", [])}
    destination = nodes.get(destination_id)
    if destination is None:
        return [f"unknown destination node {destination_id}"]

    pairs: set[tuple[str, str]] = set()
    for edge in edges:
        source_id = edge.get("source_node_id")
        actual_destination = edge.get("dest_node_id")
        relation = edge.get("label")
        if actual_destination != destination_id:
            errors.append(
                f"edge targets {actual_destination!r}, expected {destination_id!r}"
            )
            continue
        source = nodes.get(source_id)
        if source is None:
            errors.append(f"edge references unknown source {source_id!r}")
            continue
        try:
            if numeric_id(source_id) >= numeric_id(destination_id):
                errors.append(f"edge must point left-to-right: {source_id}->{destination_id}")
        except ValueError as exc:
            errors.append(str(exc))
        pair = (source_id, destination_id)
        if pair in pairs:
            errors.append(f"duplicate edge pair: {source_id}->{destination_id}")
        pairs.add(pair)
        if relation not in EDGE_RULES:
            errors.append(f"unknown edge label {relation!r}")
            continue
        allowed_sources, allowed_destinations = EDGE_RULES[relation]
        if source.get("label") not in allowed_sources:
            errors.append(
                f"{source_id} has role {source.get('label')!r}, invalid for {relation}; "
                f"allowed source roles: {sorted(allowed_sources)}"
            )
        if destination.get("label") not in allowed_destinations:
            errors.append(
                f"{destination_id} has role {destination.get('label')!r}, invalid for "
                f"{relation}; allowed destination roles: {sorted(allowed_destinations)}"
            )
    return errors


def invalid_destination_ids(doc: dict) -> list[str]:
    """Return response destinations having at least one invalid incoming edge."""
    grouped: dict[str, list[dict]] = {}
    for edge in doc.get("edges", []):
        destination_id = edge.get("dest_node_id")
        if isinstance(destination_id, str):
            grouped.setdefault(destination_id, []).append(edge)
    invalid = [
        destination_id
        for destination_id, edges in grouped.items()
        if validate_incoming_edges(doc, destination_id, edges)
    ]
    return sorted(invalid, key=numeric_id)


def replace_incoming_edges(doc: dict, destination_id: str, replacements: list[dict]) -> None:
    kept = [
        edge for edge in doc.get("edges", [])
        if edge.get("dest_node_id") != destination_id
    ]
    edges = kept + replacements
    edges.sort(
        key=lambda edge: (
            numeric_id(edge["dest_node_id"]), numeric_id(edge["source_node_id"])
        )
    )
    for index, edge in enumerate(edges):
        edge["id"] = f"e{index}"
    doc["edges"] = edges


def atomic_write(path: Path, value: dict) -> None:
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, delete=False, suffix=".json", encoding="utf-8"
    ) as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def feedback_text(errors: list[str]) -> str:
    return "Previous validation errors:\n" + "\n".join(f"- {error}" for error in errors)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotations-dir", type=Path,
        default=DEFAULT_OUT / "reasoning_annotations",
    )
    parser.add_argument(
        "--backup-dir", type=Path,
        help="Backup directory (default: OUT_ROOT/backup/edge_retries)",
    )
    parser.add_argument("--debug-root", type=Path, default=DEFAULT_OUT / "debug" / "edge_retries")
    parser.add_argument("--provider-profile", default="uvarc")
    parser.add_argument("--model", default="Kimi K2.5")
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0,
                        help="Maximum invalid destination nodes to process; 0 means all")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.attempts < 1 or args.max_tokens < 1 or args.limit < 0:
        parser.error("attempts/max-tokens must be positive and limit nonnegative")
    if not args.annotations_dir.is_dir():
        parser.error(f"annotations directory not found: {args.annotations_dir}")

    backup_dir = args.backup_dir or args.annotations_dir.parent / "backup" / "edge_retries"
    files = sorted(args.annotations_dir.glob("*.json"))
    work = []
    for path in files:
        doc = read_json(path)
        for destination_id in invalid_destination_ids(doc):
            work.append((path, destination_id))
    if args.limit:
        work = work[:args.limit]
    print(f"Found {len(work)} invalid destination node(s) across {len(files)} document(s).")
    if args.dry_run or not work:
        for path, destination_id in work:
            print(f"would retry: {path.name} {destination_id}")
        return

    rf.configure_llm(
        profile=args.provider_profile, model=args.model,
        max_tokens=args.max_tokens, debug_root=args.debug_root,
    )
    completed = failed = 0
    backed_up: set[Path] = set()
    for path, destination_id in work:
        doc = read_json(path)
        # A prior repair in the same document can change edge IDs but not node IDs.
        if destination_id not in invalid_destination_ids(doc):
            continue
        nodes = doc["nodes"]
        node_ids = [node["id"] for node in nodes]
        if destination_id not in node_ids:
            print(f"FAIL: {path.name} {destination_id}: destination disappeared", flush=True)
            failed += 1
            continue
        node_index = node_ids.index(destination_id)
        feedback = None
        repaired = False
        for attempt in range(1, args.attempts + 1):
            try:
                replacements = rf.edge_detection_and_classification(
                    node_index, nodes, args.model, validation_feedback=feedback
                )
                errors = validate_incoming_edges(doc, destination_id, replacements)
            except Exception as exc:
                errors = [f"{type(exc).__name__}: {exc}"]
            if not errors:
                if path not in backed_up:
                    backup_dir.mkdir(parents=True, exist_ok=True)
                    backup = backup_dir / path.name
                    if not backup.exists():
                        shutil.copy2(path, backup)
                    backed_up.add(path)
                replace_incoming_edges(doc, destination_id, replacements)
                atomic_write(path, doc)
                completed += 1
                repaired = True
                print(
                    f"ok: {path.name} {destination_id} "
                    f"({len(replacements)} replacement edge(s), attempt {attempt})",
                    flush=True,
                )
                break
            feedback = feedback_text(errors)
            print(
                f"retry: {path.name} {destination_id} attempt {attempt}: "
                + "; ".join(errors),
                flush=True,
            )
        if not repaired:
            failed += 1
            print(f"FAIL: {path.name} {destination_id}: exhausted retries", flush=True)

    print(f"completed={completed} failed={failed} backups={backup_dir}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
