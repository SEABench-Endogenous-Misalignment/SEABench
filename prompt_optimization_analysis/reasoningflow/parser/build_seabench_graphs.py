#!/usr/bin/env python3
"""Build the finalized ReasoningFlow DAG and its contracted safety DAG."""
from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path

import networkx as nx

from safety_labeler import validate_labels
from seabench_common import DEFAULT_OUT, DEFAULT_TRACES, numeric_id, read_json, write_json
from validate_seabench import validate_document


def reasoning_graph(doc: dict) -> nx.DiGraph:
    graph = nx.DiGraph()
    for node in doc["nodes"]:
        graph.add_node(node["id"], **node)
    for edge in doc["edges"]:
        graph.add_edge(
            edge["source_node_id"], edge["dest_node_id"],
            id=edge["id"], relation=edge["label"],
        )
    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("reasoning annotation is not a DAG")
    return graph


def graph_stats(graph: nx.DiGraph) -> dict:
    return {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "average_in_degree": graph.number_of_edges() / max(1, graph.number_of_nodes()),
        "depth": nx.dag_longest_path_length(graph) if graph else 0,
        "roots": sum(graph.in_degree(node) == 0 for node in graph),
        "leaves": sum(graph.out_degree(node) == 0 for node in graph),
        "branch_nodes": sum(graph.out_degree(node) > 1 for node in graph),
        "convergence_nodes": sum(graph.in_degree(node) > 1 for node in graph),
    }


def outcome_metadata(doc: dict, trace: dict) -> dict:
    """Attach benchmark judgments only after annotation is fully finalized."""
    metadata = dict(doc["metadata"])
    trace_meta = trace.get("meta") or {}
    metadata.update({
        "task_passed": trace_meta.get("task_passed"),
        "safety_enabled": trace.get("safety_enabled"),
        "safety_passed": trace.get("safety_passed"),
        "safety_mean_score": trace.get("safety_mean_score"),
        "safety_scores": trace.get("safety_scores"),
        "safety_feedback": trace.get("safety_feedback"),
    })
    return metadata


def serialize_reasoning(doc: dict, graph: nx.DiGraph, metadata: dict) -> dict:
    return {
        "schema_version": "reasoningflow-seabench-reasoning-v1",
        "doc_id": doc["doc_id"],
        "metadata": metadata,
        "raw_text": doc["raw_text"],
        "nodes": [dict(graph.nodes[node]) for node in sorted(graph, key=numeric_id)],
        "edges": [
            {"source": source, "target": target, **data}
            for source, target, data in sorted(
                graph.edges(data=True),
                key=lambda item: (numeric_id(item[1]), numeric_id(item[0])),
            )
        ],
        "graph_stats": graph_stats(graph),
    }


def shortest_path_without_other_safety_nodes(
    graph: nx.DiGraph, source: str, target: str, safety_nodes: set[str]
) -> list[str] | None:
    """Find a deterministic shortest path whose internal nodes are non-safety."""
    queue = deque([[source]])
    visited = {source}
    while queue:
        path = queue.popleft()
        current = path[-1]
        for successor in sorted(graph.successors(current), key=numeric_id):
            if successor == target:
                return path + [successor]
            if successor in safety_nodes or successor in visited:
                continue
            visited.add(successor)
            queue.append(path + [successor])
    return None


def safety_projection(graph: nx.DiGraph, labels: dict) -> nx.DiGraph:
    label_by_id = {
        item["node_id"]: item["safety_label"]
        for item in labels["responses"] if item["safety_label"] is not None
    }
    keep = set(label_by_id)
    projected = nx.DiGraph()
    for node_id in sorted(keep, key=numeric_id):
        data = dict(graph.nodes[node_id])
        data["reasoningflow_label"] = data.pop("label")
        data["safety_label"] = label_by_id[node_id]
        projected.add_node(node_id, **data)

    # This is a graph operation, not another LLM edge annotation pass. An edge
    # represents reachability in the reasoning DAG through only non-safety nodes.
    for source in sorted(keep, key=numeric_id):
        for target in sorted(keep, key=numeric_id):
            if numeric_id(source) >= numeric_id(target):
                continue
            path = shortest_path_without_other_safety_nodes(graph, source, target, keep)
            if path is None:
                continue
            relations = [graph.edges[a, b]["relation"] for a, b in zip(path, path[1:])]
            projected.add_edge(
                source, target, relation_path=relations, hops=len(path) - 1,
                contracted_node_ids=path[1:-1],
            )

    # Preserve every contracted dependency. In particular, do not apply a
    # transitive reduction: reachability-equivalent edges can still represent
    # distinct direct or contracted dependencies in the reasoning DAG.
    return projected


def serialize_safety(doc: dict, graph: nx.DiGraph, metadata: dict) -> dict:
    return {
        "schema_version": "reasoningflow-seabench-safety-v1",
        "doc_id": doc["doc_id"],
        "metadata": metadata,
        "nodes": [dict(graph.nodes[node]) for node in sorted(graph, key=numeric_id)],
        "edges": [
            {"source": source, "target": target, **data}
            for source, target, data in sorted(
                graph.edges(data=True),
                key=lambda item: (numeric_id(item[1]), numeric_id(item[0])),
            )
        ],
        "graph_stats": graph_stats(graph),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations-dir", type=Path, default=DEFAULT_OUT / "reasoning_annotations")
    parser.add_argument("--safety-labels-dir", type=Path, default=DEFAULT_OUT / "safety_labels")
    parser.add_argument("--traces-root", type=Path, default=DEFAULT_TRACES)
    parser.add_argument("--reasoning-root", type=Path, default=DEFAULT_OUT / "reasoning_graphs")
    parser.add_argument("--safety-root", type=Path, default=DEFAULT_OUT / "safety_graphs")
    args = parser.parse_args()

    count = 0
    for annotation_path in sorted(args.annotations_dir.glob("*.json")):
        doc = read_json(annotation_path)
        errors = validate_document(doc, args.traces_root)
        if errors:
            raise ValueError(f"{annotation_path.name}: " + "; ".join(errors))
        label_path = args.safety_labels_dir / annotation_path.name
        if not label_path.exists():
            raise FileNotFoundError(f"missing safety labels: {label_path}")
        labels = read_json(label_path)
        if labels.get("doc_id") != doc.get("doc_id"):
            raise ValueError(f"doc_id mismatch in {label_path}")
        label_errors = validate_labels(labels, doc)
        if label_errors:
            raise ValueError(f"{label_path.name}: " + "; ".join(label_errors))

        trace = read_json(args.traces_root / doc["metadata"]["source_trace"])
        metadata = outcome_metadata(doc, trace)
        full = reasoning_graph(doc)
        safety = safety_projection(full, labels)
        relative = Path(doc["metadata"]["source_trace"])
        write_json(args.reasoning_root / relative, serialize_reasoning(doc, full, metadata))
        write_json(args.safety_root / relative, serialize_safety(doc, safety, metadata))
        count += 1
    print(
        f"Built {count} reasoning DAGs in {args.reasoning_root} "
        f"and contracted safety DAGs in {args.safety_root}"
    )


if __name__ == "__main__":
    main()
