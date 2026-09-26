#!/usr/bin/env python3
"""Run every ReasoningFlow and safety stage to completion one trace at a time."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
import shutil
import tempfile
from pathlib import Path

import llm_labeler as rf
from build_seabench_graphs import (
    outcome_metadata,
    reasoning_graph,
    safety_projection,
    serialize_reasoning,
    serialize_safety,
)
from cleanse_scripts import (
    cleanse_common_errors,
    cleanse_reorder_nodes,
    cleanse_think_metatags,
)
from safety_labeler import (
    configure as configure_safety,
    label_document as label_safety_document,
    normalize_surface,
    prompt_fingerprint,
    validate_labels,
)
from retry_invalid_edges import (
    feedback_text,
    invalid_destination_ids,
    replace_incoming_edges,
    validate_incoming_edges,
)
from seabench_annotate import annotate_document
from seabench_common import convert_trace, iter_trace_files, read_json, trace_has_reasoning
from validate_seabench import validate_document


def atomic_write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, delete=False, suffix=".json", encoding="utf-8"
    ) as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def document_fingerprint(doc: dict) -> str:
    payload = json.dumps(doc, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def cleanse_one(doc: dict, annotation_dir: Path) -> dict:
    """Apply the mutating original cleansers in their established order."""
    annotation_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=annotation_dir, prefix=".cleanse-") as directory:
        path = Path(directory) / "annotation.json"
        atomic_write_json(path, doc)
        # The original cleansers print implementation-specific pattern names and
        # the temporary filename. Normal pipeline output reports this as one stage.
        with contextlib.redirect_stdout(io.StringIO()):
            cleanse_think_metatags.run(dry_run=False, data_dir=directory)
            cleanse_common_errors.run(dry_run=False, pattern=1, data_dir=directory)
            cleanse_common_errors.run(dry_run=False, pattern=2, data_dir=directory)
            cleanse_reorder_nodes.process_data_file(str(path))
        return read_json(path)


def expected_graphs(doc: dict, labels: dict, trace: dict) -> tuple[dict, dict]:
    metadata = outcome_metadata(doc, trace)
    full = reasoning_graph(doc)
    safety = safety_projection(full, labels)
    return (
        serialize_reasoning(doc, full, metadata),
        serialize_safety(doc, safety, metadata),
    )


def valid_safety_labels(
    path: Path, doc: dict, annotation_fingerprint: str,
    safety_prompt_fingerprint: str, surface: str,
) -> dict | None:
    if not path.exists():
        return None
    try:
        labels = read_json(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if labels.get("doc_id") != doc.get("doc_id") or validate_labels(labels, doc):
        return None
    if labels.get("annotation_sha256") != annotation_fingerprint:
        return None
    if labels.get("safety_prompt_sha256") != safety_prompt_fingerprint:
        return None
    if labels.get("surface") != normalize_surface(surface):
        return None
    return labels


def load_valid_annotation(path: Path, traces_root: Path) -> dict | None:
    doc = load_annotation(path)
    return doc if doc is not None and not validate_document(doc, traces_root) else None


def load_annotation(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return read_json(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def repair_invalid_edges(doc: dict, model: str, attempts: int) -> bool:
    """Retry only invalid incoming-edge sets, preserving all other annotation work."""
    changed = False
    for destination_id in invalid_destination_ids(doc):
        nodes = doc["nodes"]
        node_ids = [node["id"] for node in nodes]
        if destination_id not in node_ids:
            raise ValueError(f"invalid edge destination disappeared: {destination_id}")
        node_index = node_ids.index(destination_id)
        current = [
            edge for edge in doc.get("edges", [])
            if edge.get("dest_node_id") == destination_id
        ]
        feedback = feedback_text(validate_incoming_edges(doc, destination_id, current))
        for attempt in range(1, attempts + 1):
            try:
                replacements = rf.edge_detection_and_classification(
                    node_index, nodes, model, validation_feedback=feedback
                )
                errors = validate_incoming_edges(doc, destination_id, replacements)
            except Exception as exc:
                replacements = []
                errors = [f"{type(exc).__name__}: {exc}"]
            if not errors:
                replace_incoming_edges(doc, destination_id, replacements)
                changed = True
                print(
                    f"edge repair: {destination_id} "
                    f"({len(replacements)} replacement edge(s), attempt {attempt})",
                    flush=True,
                )
                break
            feedback = feedback_text(errors)
            print(
                f"edge retry: {destination_id} attempt {attempt}: " + "; ".join(errors),
                flush=True,
            )
        else:
            raise ValueError(
                f"{destination_id}: invalid incoming edges after {attempts} attempts"
            )
    return changed


def backup_annotation(path: Path, backup_dir: Path) -> Path:
    backup_dir.mkdir(parents=True, exist_ok=True)
    destination = backup_dir / path.name
    if not destination.exists():
        shutil.copy2(path, destination)
    return destination


def graph_file_matches(path: Path, expected: dict) -> bool:
    if not path.exists():
        return False
    try:
        return read_json(path) == expected
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def trace_is_complete(
    source: Path, traces_root: Path, annotation_path: Path, safety_path: Path,
    reasoning_path: Path, safety_graph_path: Path, surface: str,
    reasoningflow_only: bool = False,
) -> bool:
    doc = load_valid_annotation(annotation_path, traces_root)
    if doc is None:
        return False
    trace = read_json(source)
    if reasoningflow_only:
        expected = serialize_reasoning(
            doc, reasoning_graph(doc), outcome_metadata(doc, trace)
        )
        return graph_file_matches(reasoning_path, expected)
    fingerprint = document_fingerprint(doc)
    labels = valid_safety_labels(
        safety_path, doc, fingerprint, prompt_fingerprint(doc, trace, surface), surface
    )
    if labels is None:
        return False
    expected_reasoning, expected_safety = expected_graphs(doc, labels, trace)
    return (
        graph_file_matches(reasoning_path, expected_reasoning)
        and graph_file_matches(safety_graph_path, expected_safety)
    )


def preflight_status(
    source: Path, traces_root: Path, annotation_path: Path, safety_path: Path,
    reasoning_path: Path, safety_graph_path: Path, surface: str,
    reasoningflow_only: bool = False,
) -> str:
    if trace_is_complete(
        source, traces_root, annotation_path, safety_path,
        reasoning_path, safety_graph_path, surface, reasoningflow_only,
    ):
        return "complete"
    doc = load_annotation(annotation_path)
    if doc is None:
        return "reasoning"
    errors = validate_document(doc, traces_root)
    if errors:
        return "edge_repair" if invalid_destination_ids(doc) else "cleanse_or_validate"
    if reasoningflow_only:
        return "graphs"
    labels = valid_safety_labels(
        safety_path, doc, document_fingerprint(doc),
        prompt_fingerprint(doc, read_json(source), surface), surface,
    )
    if labels is None:
        return "safety"
    return "graphs"


def print_preflight(
    sources: list[Path], traces_root: Path, annotations_dir: Path,
    safety_labels_dir: Path, reasoning_root: Path, safety_root: Path, surface: str,
    reasoningflow_only: bool = False,
) -> None:
    statuses: dict[str, list[str]] = {}
    for source in sources:
        rel = source.relative_to(traces_root)
        filename = "__".join(rel.with_suffix("").parts) + ".json"
        status = preflight_status(
            source, traces_root, annotations_dir / filename,
            safety_labels_dir / filename, reasoning_root / rel, safety_root / rel, surface,
            reasoningflow_only,
        )
        statuses.setdefault(status, []).append(rel.as_posix())
    order = ("complete", "graphs", "safety", "edge_repair", "cleanse_or_validate", "reasoning")
    print("Per-trace resume preflight (no files changed):")
    for status in order:
        members = statuses.get(status, [])
        print(f"  {status}: {len(members)}")
        for relative in members if status != "complete" else []:
            print(f"    {relative}")


def validate_complete_dataset(
    sources: list[Path], traces_root: Path, annotations_dir: Path,
    safety_labels_dir: Path, reasoning_root: Path, safety_root: Path, surface: str,
    reasoningflow_only: bool = False,
) -> None:
    errors = []
    for source in sources:
        rel = source.relative_to(traces_root)
        filename = "__".join(rel.with_suffix("").parts) + ".json"
        if not trace_is_complete(
            source, traces_root, annotations_dir / filename, safety_labels_dir / filename,
            reasoning_root / rel, safety_root / rel, surface, reasoningflow_only,
        ):
            errors.append(rel.as_posix())
    if errors:
        preview = ", ".join(errors[:5])
        suffix = "" if len(errors) <= 5 else f", and {len(errors) - 5} more"
        raise ValueError(f"completeness failure for {len(errors)} trace(s): {preview}{suffix}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--reasoning-debug-root", type=Path, required=True)
    parser.add_argument("--edge-debug-root", type=Path, required=True)
    parser.add_argument("--safety-debug-root", type=Path, required=True)
    parser.add_argument("--provider-profile", default="uvarc")
    parser.add_argument("--model", default="Kimi K2.5")
    parser.add_argument(
        "--surface", required=True,
        choices=("short_term_memory", "controller_update", "tool-use"),
    )
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--edge-retry-attempts", type=int, default=3)
    parser.add_argument("--node-retry-attempts", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--preflight", action="store_true",
                        help="Report resume actions without changing files or calling an LLM")
    parser.add_argument(
        "--safety-only", action="store_true",
        help=(
            "Force a fresh safety annotation and rebuild only the safety graph, using the "
            "existing finalized reasoning annotation; never cleanse, repair, or regenerate "
            "reasoning annotations"
        ),
    )
    parser.add_argument(
        "--reasoningflow-only", action="store_true",
        help=(
            "Run only the ReasoningFlow node and edge annotation (plus cleansing, edge "
            "repair, and the reasoning graph); skip safety annotation and safety graphs"
        ),
    )
    args = parser.parse_args()
    if args.reasoningflow_only and args.safety_only:
        parser.error("--reasoningflow-only and --safety-only are mutually exclusive")
    if (args.workers < 1 or args.edge_retry_attempts < 1 or args.node_retry_attempts < 1
            or args.max_tokens < 1 or args.limit < 0):
        parser.error(
            "workers/edge-retry-attempts/node-retry-attempts/max-tokens must be positive "
            "and limit nonnegative"
        )

    annotations_dir = args.out_root / "reasoning_annotations"
    safety_labels_dir = args.out_root / "safety_labels"
    reasoning_root = args.out_root / "reasoning_graphs"
    safety_root = args.out_root / "safety_graphs"
    all_sources = list(iter_trace_files(args.traces_root))
    if not all_sources:
        raise SystemExit(f"No trace files found under {args.traces_root}")
    sources = []
    excluded_sources = []
    for source in all_sources:
        if trace_has_reasoning(read_json(source)):
            sources.append(source)
        else:
            excluded_sources.append(source)
    if excluded_sources:
        print(
            f"Excluding {len(excluded_sources)} trace(s) with no generated reasoning "
            "(not annotated, not required for completeness):",
            flush=True,
        )
        for source in excluded_sources:
            print(f"  {source.relative_to(args.traces_root)}", flush=True)
    if args.preflight:
        print_preflight(
            sources, args.traces_root, annotations_dir, safety_labels_dir,
            reasoning_root, safety_root, args.surface, args.reasoningflow_only,
        )
        return

    attempted = completed = skipped = failed = 0
    printed_trace = False
    for source in sources:
        rel = source.relative_to(args.traces_root)
        filename = "__".join(rel.with_suffix("").parts) + ".json"
        annotation_path = annotations_dir / filename
        safety_path = safety_labels_dir / filename
        reasoning_path = reasoning_root / rel
        safety_graph_path = safety_root / rel

        if args.safety_only:
            if args.limit and attempted >= args.limit:
                break
            attempted += 1
            if printed_trace:
                print()
            print(f"safety-only processing: {rel}", flush=True)
            printed_trace = True
            try:
                trace = read_json(source)
                doc = load_valid_annotation(annotation_path, args.traces_root)
                if doc is None:
                    skipped += 1
                    attempted -= 1
                    print(
                        "Skipping: no valid finalized reasoning annotation (safety-only mode "
                        f"will not regenerate it): {rel}", flush=True,
                    )
                    continue
                configure_safety(
                    profile=args.provider_profile, model=args.model,
                    max_tokens=args.max_tokens, debug_root=args.safety_debug_root,
                )
                labels = label_safety_document(doc, trace, args.model, args.surface)
                labels["annotation_sha256"] = document_fingerprint(doc)
                if safety_path.exists():
                    backup_annotation(
                        safety_path, args.out_root / "backup" / "safety_labels"
                    )
                    print(f"Backed up prior safety annotation: {rel}", flush=True)
                atomic_write_json(safety_path, labels)
                metadata = outcome_metadata(doc, trace)
                safety = safety_projection(reasoning_graph(doc), labels)
                atomic_write_json(safety_graph_path, serialize_safety(doc, safety, metadata))
                completed += 1
                print(f"Safety annotation and safety graph complete: {rel}", flush=True)
            except Exception as exc:
                failed += 1
                print(f"FAIL: {rel}: {type(exc).__name__}: {exc}", flush=True)
            continue

        if trace_is_complete(
            source, args.traces_root, annotation_path, safety_path,
            reasoning_path, safety_graph_path, args.surface, args.reasoningflow_only,
        ):
            skipped += 1
            if printed_trace:
                print()
            print(f"Skipping complete trace: {rel}", flush=True)
            printed_trace = True
            continue
        if args.limit and attempted >= args.limit:
            break
        attempted += 1
        if printed_trace:
            print()
        print(f"processing: {rel}", flush=True)
        printed_trace = True

        try:
            trace = read_json(source)
            # A saved annotation from the former batch pipeline may be awaiting its
            # cleansing/validation stages. Preserve it and repair only bad edge sets.
            doc = load_annotation(annotation_path)
            annotation_rebuilt = doc is None
            if doc is None:
                print(
                    f"Reasoning annotation missing; running full annotation: {rel}",
                    flush=True,
                )
                rf.configure_llm(
                    profile=args.provider_profile, model=args.model,
                    max_tokens=args.max_tokens, debug_root=args.reasoning_debug_root,
                )
                doc = annotate_document(convert_trace(trace, rel), args.model, args.workers,
                                        node_retry_attempts=args.node_retry_attempts)
                print(f"Reasoning annotation complete: {rel}", flush=True)
            else:
                print(f"Reasoning annotation present: {rel}", flush=True)

            print(f"Cleansing reasoning annotation: {rel}", flush=True)
            cleansed = cleanse_one(doc, annotations_dir)
            print(f"Cleansing complete: {rel}", flush=True)
            rf.configure_llm(
                profile=args.provider_profile, model=args.model,
                max_tokens=args.max_tokens, debug_root=args.edge_debug_root,
            )
            invalid_edges = invalid_destination_ids(cleansed)
            if invalid_edges:
                print(
                    f"Repairing {len(invalid_edges)} invalid edge destination(s): {rel}",
                    flush=True,
                )
                edges_repaired = repair_invalid_edges(
                    cleansed, args.model, args.edge_retry_attempts
                )
                print(f"Edge repair complete: {rel}", flush=True)
            else:
                edges_repaired = False
                print(f"Edge validation passed: {rel}", flush=True)
            errors = validate_document(cleansed, args.traces_root)
            if errors:
                raise ValueError(f"{rel}: " + "; ".join(errors))
            changed = cleansed != doc
            doc = cleansed
            if annotation_rebuilt or changed or edges_repaired or not annotation_path.exists():
                if edges_repaired and annotation_path.exists():
                    backup_annotation(
                        annotation_path, args.out_root / "backup" / "edge_retries"
                    )
                atomic_write_json(annotation_path, doc)

            if args.reasoningflow_only:
                print(f"Building reasoning graph (safety skipped): {rel}", flush=True)
                atomic_write_json(reasoning_path, serialize_reasoning(
                    doc, reasoning_graph(doc), outcome_metadata(doc, trace)
                ))
                completed += 1
                print(f"Completed trace: {rel}", flush=True)
                continue

            fingerprint = document_fingerprint(doc)
            safety_fingerprint = prompt_fingerprint(doc, trace, args.surface)
            labels = valid_safety_labels(
                safety_path, doc, fingerprint, safety_fingerprint, args.surface
            )
            if labels is None:
                print(f"Running safety annotation: {rel}", flush=True)
                configure_safety(
                    profile=args.provider_profile, model=args.model,
                    max_tokens=args.max_tokens, debug_root=args.safety_debug_root,
                )
                labels = label_safety_document(doc, trace, args.model, args.surface)
                labels["annotation_sha256"] = fingerprint
                if safety_path.exists():
                    backup_annotation(
                        safety_path, args.out_root / "backup" / "safety_labels"
                    )
                    print(f"Backed up stale safety annotation: {rel}", flush=True)
                atomic_write_json(safety_path, labels)
                print(f"Safety annotation complete: {rel}", flush=True)
            else:
                print(f"Safety annotation present and valid: {rel}", flush=True)

            print(f"Building reasoning and safety graphs: {rel}", flush=True)
            expected_reasoning, expected_safety = expected_graphs(doc, labels, trace)
            atomic_write_json(reasoning_path, expected_reasoning)
            atomic_write_json(safety_graph_path, expected_safety)
            completed += 1
            print(f"Completed trace: {rel}", flush=True)
        except Exception as exc:
            failed += 1
            print(f"FAIL: {rel}: {type(exc).__name__}: {exc}", flush=True)

    if args.limit == 0 and not args.safety_only:
        validate_complete_dataset(
            sources, args.traces_root, annotations_dir, safety_labels_dir,
            reasoning_root, safety_root, args.surface, args.reasoningflow_only,
        )
    print(
        f"attempted={attempted} completed={completed} skipped={skipped} "
        f"failed={failed} excluded={len(excluded_sources)}"
    )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
