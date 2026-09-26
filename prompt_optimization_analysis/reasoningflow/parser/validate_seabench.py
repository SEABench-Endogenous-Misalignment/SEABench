#!/usr/bin/env python3
"""Validate finalized upstream-style ReasoningFlow annotations for SEABench."""
from __future__ import annotations

import argparse
from pathlib import Path

import networkx as nx
import yaml

from seabench_common import (
    DEFAULT_OUT, DEFAULT_TRACES, RF_ROOT, convert_trace, numeric_id, read_json,
)

NODE_SCHEMA = yaml.safe_load((RF_ROOT / "schema/node_labels.yaml").read_text())
EDGE_SCHEMA = yaml.safe_load((RF_ROOT / "schema/edge_labels.yaml").read_text())
NODE_LABELS = {x["name"] for x in NODE_SCHEMA["nodes"]}
EDGE_RULES = {
    x["name"]: (set(x.get("source", [])), set(x.get("dest", [])))
    for x in EDGE_SCHEMA["edges"]
}


def validate_document(doc: dict, traces_root: Path) -> list[str]:
    errors = []
    meta = doc.get("metadata") or {}
    source_rel = meta.get("source_trace")
    relative = Path(source_rel) if isinstance(source_rel, str) else None
    if relative is not None and (relative.is_absolute() or ".." in relative.parts):
        errors.append("metadata.source_trace must be a relative path within traces root")
        relative = None
    source = traces_root / relative if relative is not None else None
    if source is None or not source.exists():
        errors.append("metadata.source_trace does not resolve to an extracted trace")
    forbidden = {"answer", "safety_passed", "safety_scores", "safety_feedback", "safety_mean_score"}
    leaked = forbidden & (set(meta) | set(doc))
    if leaked:
        errors.append(f"outcome/answer fields leaked into annotation document: {sorted(leaked)}")
    raw = doc.get("raw_text") or {}
    if source is not None and source.exists():
        expected = convert_trace(read_json(source), relative)
        if raw != expected["raw_text"]:
            errors.append("raw_text differs from the answer-free source conversion")
        if meta.get("arm") != expected["metadata"].get("arm"):
            errors.append("metadata.arm differs from source trace")
        if doc.get("doc_id") != expected["doc_id"]:
            errors.append("doc_id differs from source trace path")
    nodes = doc.get("nodes")
    edges = doc.get("edges")
    if not isinstance(nodes, list) or not nodes:
        return errors + ["nodes must be a non-empty list"]
    if not isinstance(edges, list):
        return errors + ["edges must be a list"]
    by_id = {}
    by_source = {"question": [], "response": []}
    previous_key = (-1, -1)
    for i, node in enumerate(nodes):
        node_id = node.get("id")
        try:
            key = numeric_id(str(node_id))
        except ValueError as exc:
            errors.append(str(exc)); continue
        if key < previous_key:
            errors.append(f"node order goes backward at {node_id}")
        previous_key = key
        if node_id in by_id:
            errors.append(f"duplicate node id {node_id}")
        by_id[node_id] = node
        source_name = node.get("source")
        if source_name not in by_source:
            errors.append(f"{node_id}: invalid source {source_name!r}"); continue
        start, end = node.get("start"), node.get("end")
        if not isinstance(start, int) or not isinstance(end, int) or not 0 <= start < end:
            errors.append(f"{node_id}: invalid offsets"); continue
        expected = str(raw.get(source_name, ""))[start:end]
        if node.get("text") != expected:
            errors.append(f"{node_id}: text does not equal its source slice")
        by_source[source_name].append(node)
        label = node.get("label")
        if label not in NODE_LABELS:
            errors.append(f"{node_id}: invalid label {label!r}")
        if source_name == "question" and label != "context":
            errors.append(f"{node_id}: question nodes must be context")
        if source_name == "response" and label == "context":
            errors.append(f"{node_id}: response node cannot be context")
    for source_name, source_nodes in by_source.items():
        reconstructed = "".join(n.get("text", "") for n in source_nodes)
        if reconstructed != str(raw.get(source_name, "")):
            errors.append(f"{source_name}: nodes do not reconstruct raw text")
    graph = nx.DiGraph()
    graph.add_nodes_from(by_id)
    pairs = set()
    for edge in edges:
        source_id, target_id, relation = (
            edge.get("source_node_id"), edge.get("dest_node_id"), edge.get("label")
        )
        if source_id not in by_id or target_id not in by_id:
            errors.append(f"edge references unknown node: {source_id}->{target_id}"); continue
        if numeric_id(source_id) >= numeric_id(target_id):
            errors.append(f"edge must point left-to-right: {source_id}->{target_id}")
        if (source_id, target_id) in pairs:
            errors.append(f"multiple edges for node pair: {source_id}->{target_id}")
        pairs.add((source_id, target_id))
        if relation not in EDGE_RULES:
            errors.append(f"invalid edge relation {relation!r}"); continue
        allowed_source, allowed_target = EDGE_RULES[relation]
        if by_id[source_id]["label"] not in allowed_source:
            errors.append(f"{relation}: invalid source role {by_id[source_id]['label']}")
        if by_id[target_id]["label"] not in allowed_target:
            errors.append(f"{relation}: invalid destination role {by_id[target_id]['label']}")
        graph.add_edge(source_id, target_id)
    if not nx.is_directed_acyclic_graph(graph):
        errors.append("graph is not acyclic")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations-dir", type=Path, default=DEFAULT_OUT / "reasoning_annotations")
    parser.add_argument("--traces-root", type=Path, default=DEFAULT_TRACES)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    files = sorted(args.annotations_dir.glob("*.json"))
    errors = []
    for path in files:
        for error in validate_document(read_json(path), args.traces_root):
            errors.append(f"{path.name}: {error}")
    if args.require_complete:
        expected = sum(1 for _ in args.traces_root.glob("*/*/*/*.json"))
        if len(files) != expected:
            errors.append(f"completeness failure: {len(files)} annotations for {expected} traces")
    if errors:
        print("\n".join(errors))
        raise SystemExit(f"validation failed with {len(errors)} error(s)")
    print(f"Validated {len(files)} ReasoningFlow annotations")


if __name__ == "__main__":
    main()
