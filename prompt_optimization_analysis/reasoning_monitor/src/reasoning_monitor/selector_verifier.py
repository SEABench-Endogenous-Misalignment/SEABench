#!/usr/bin/env python3
"""Train, validate, freeze, and test the selector and semantic verifier."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import random
import re
import shutil
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler


SOURCE = Path(__file__).resolve().parent
THREAD_NODE_FIELDS = (
    "boundary_node_ids", "risk_node_ids", "plan_node_ids",
    "correction_node_ids", "reopening_node_ids",
)


@dataclass
class Node:
    node_id: str
    unit: int
    text: str
    rf_label: str
    annotations: list[dict[str, Any]]


@dataclass
class RiskThread:
    instance_id: str
    trace_id: str
    risk_id: str
    object_id: str
    object_record: dict[str, Any]
    record: dict[str, Any]
    text: str
    features: dict[str, float]
    warning_unit: int
    current_unit: int
    max_unit: int


@dataclass
class Trace:
    trace_id: str
    group: str
    task: str
    outcome: str
    surface: str
    arm: str
    nodes: list[Node]
    relations: list[dict[str, Any]]
    risk_objects: dict[str, dict[str, Any]]
    risk_threads: list[dict[str, Any]]
    instances: list[RiskThread]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verifier_prompt_path() -> Path:
    configured = os.environ.get("REASONING_MONITOR_PROMPT", "").strip()
    path = Path(configured).resolve() if configured else SOURCE / "verifier_prompt.md"
    if not path.is_file():
        raise FileNotFoundError(f"verifier prompt does not exist: {path}")
    return path


def clean_task(value: str) -> str:
    return re.sub(r"\A\s*##[^\n]*\n+", "", str(value or "")).strip()


def unit(node_id: str) -> int:
    match = re.fullmatch(r"resp(\d+)", node_id)
    if not match:
        raise ValueError(f"unexpected response node ID: {node_id}")
    return int(match.group(1))


def surface_name(directory: str) -> str:
    return {
        "controller_update": "controller",
        "short_term_memory": "memory",
        "tools_skills": "tools_skills",
    }[directory]


def normalized_group_harm(value: str) -> str:
    # Controller uses the longer directory label for the same benchmark family.
    return "boundary_collapse" if value == "contextual_boundary_collapse" else value


def make_thread_instance(
    trace_id: str,
    task: str,
    nodes: list[Node],
    relations: list[dict[str, Any]],
    object_record: dict[str, Any],
    thread: dict[str, Any],
) -> RiskThread:
    by_id = {node.node_id: node for node in nodes}
    object_id = str(thread["object_id"])
    node_ids = {
        str(node_id)
        for field in THREAD_NODE_FIELDS
        for node_id in thread.get(field, [])
    }
    node_ids.update(
        node.node_id
        for node in nodes
        if any(str(row.get("object_id")) == object_id for row in node.annotations)
    )
    relevant_nodes = [node for node in nodes if node.node_id in node_ids]
    relevant_relations = [
        row for row in relations if str(row.get("object_id")) == object_id
    ]
    features: dict[str, float] = {
        "bias": 1.0,
        f"state={thread['final_state']}": 1.0,
        f"mechanism={thread['mechanism']}": 1.0,
        f"support={thread['support_visibility']}": 1.0,
        f"authorization={thread['authorization']}": 1.0,
        f"necessity={thread['necessity']}": 1.0,
        f"object_kind={object_record['kind']}": 1.0,
        f"state+mechanism={thread['final_state']}+{thread['mechanism']}": 1.0,
        f"authorization+necessity={thread['authorization']}+{thread['necessity']}": 1.0,
        "risk_node_count": float(len(thread.get("risk_node_ids", []))),
        "plan_node_count": float(len(thread.get("plan_node_ids", []))),
        "correction_node_count": float(len(thread.get("correction_node_ids", []))),
        "reopening_node_count": float(len(thread.get("reopening_node_ids", []))),
    }
    max_unit = max((node.unit for node in nodes), default=1)
    risk_units = [unit(value) for value in thread.get("risk_node_ids", [])]
    plan_units = [unit(value) for value in thread.get("plan_node_ids", [])]
    correction_units = [unit(value) for value in thread.get("correction_node_ids", [])]
    reopening_units = [unit(value) for value in thread.get("reopening_node_ids", [])]
    boundary_units = [unit(value) for value in thread.get("boundary_node_ids", [])]
    warning_unit = min(risk_units or boundary_units or plan_units or [max_unit])
    current_unit = max(
        risk_units + plan_units + correction_units + reopening_units + boundary_units
        or [warning_unit]
    )
    features["warning_position"] = warning_unit / max_unit
    features["current_position"] = current_unit / max_unit
    features["thread_span"] = (current_unit - warning_unit) / max_unit

    events: dict[int, set[str]] = defaultdict(set)
    for node in relevant_nodes:
        features[f"rf={node.rf_label}"] = features.get(f"rf={node.rf_label}", 0.0) + 1.0
        for row in node.annotations:
            if str(row.get("object_id")) != object_id:
                continue
            for predicate in map(str, row.get("predicates", [])):
                features[f"predicate={predicate}"] = features.get(
                    f"predicate={predicate}", 0.0
                ) + 1.0
                events[node.unit].add(f"node:{predicate}")
    for relation in relevant_relations:
        predicate = str(relation["predicate"])
        features[f"relation={predicate}"] = features.get(
            f"relation={predicate}", 0.0
        ) + 1.0
        events[unit(str(relation["target_node_id"]))].add(f"relation:{predicate}")
    ordered = [sorted(events[position]) for position in sorted(events)]
    for left, right in zip(ordered, ordered[1:]):
        for first, second in itertools.product(left, right):
            if first != second:
                features[f"ordered2={first}->{second}"] = 1.0
    for left, middle, right in zip(ordered, ordered[1:], ordered[2:]):
        for values in itertools.product(left, middle, right):
            if len(set(values)) >= 2:
                features[f"ordered3={'->'.join(values)}"] = 1.0

    node_text = "\n".join(
        f"[{node.node_id} RF={node.rf_label}] {node.text}" for node in relevant_nodes
    )
    relation_text = "\n".join(
        f"[{','.join(map(str, row['source_node_ids']))}] --{row['predicate']}--> "
        f"{row['target_node_id']}: {row.get('rationale', '')}"
        for row in relevant_relations
    )
    text = (
        f"TASK:\n{task}\n\nOBJECT:\n{object_record.get('description', '')}\n"
        f"KIND: {object_record.get('kind')}\nPURPOSE: {object_record.get('intended_purpose')}\n"
        f"RECIPIENT: {object_record.get('intended_recipient')}\n"
        f"BOUNDARY:\n{thread.get('boundary', '')}\n"
        f"THREAD RATIONALE:\n{thread.get('rationale', '')}\n"
        f"STATE: {thread.get('final_state')}\nMECHANISM: {thread.get('mechanism')}\n\n"
        f"REASONING:\n{node_text}\n\nRELATIONS:\n{relation_text}"
    )
    return RiskThread(
        instance_id=f"{trace_id}::{thread['risk_id']}",
        trace_id=trace_id,
        risk_id=str(thread["risk_id"]),
        object_id=object_id,
        object_record=object_record,
        record=thread,
        text=text,
        features=features,
        warning_unit=warning_unit,
        current_unit=current_unit,
        max_unit=max_unit,
    )


def load_corpus(root: Path, config: dict[str, Any]) -> list[Trace]:
    candidate_states = set(map(str, config["candidate_states"]))
    traces = []
    for surface_dir in ("controller_update", "short_term_memory", "tools_skills"):
        surface_root = root / surface_dir
        graph_root = surface_root / "reasoning_graphs"
        for graph_path in sorted(graph_root.rglob("*.json")):
            relative = graph_path.relative_to(graph_root)
            annotation_path = surface_root / "risk_thread_safety_labels_v2" / relative
            if not annotation_path.is_file():
                raise ValueError(f"missing risk-thread annotation: {annotation_path}")
            graph = read_json(graph_path)
            annotation = read_json(annotation_path)
            meta = graph["metadata"]
            task = clean_task(graph.get("raw_text", {}).get("question", ""))
            trace_id = f"{surface_name(surface_dir)}/{relative.with_suffix('')}"
            group = (
                f"{meta['category']}/{normalized_group_harm(str(meta['harmtype']))}/"
                f"{meta['task_id']}"
            )
            node_annotations: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in annotation.get("node_annotations", []):
                node_annotations[str(row["node_id"])].append(row)
            nodes = [
                Node(
                    node_id=str(row["id"]),
                    unit=unit(str(row["id"])),
                    text=str(row.get("text", "")).strip(),
                    rf_label=str(row.get("label") or "NONE"),
                    annotations=node_annotations.get(str(row["id"]), []),
                )
                for row in graph.get("nodes", [])
                if row.get("source") == "response"
            ]
            objects = {
                str(row["object_id"]): row for row in annotation.get("risk_objects", [])
            }
            relations = list(annotation.get("relations", []))
            instances = [
                make_thread_instance(
                    trace_id, task, nodes, relations, objects[str(thread["object_id"])], thread
                )
                for thread in annotation.get("risk_threads", [])
                if str(thread.get("final_state")) in candidate_states
            ]
            traces.append(Trace(
                trace_id=trace_id,
                group=group,
                task=task,
                outcome="safe" if bool(meta["safety_passed"]) else "failure",
                surface=surface_name(surface_dir),
                arm=str(meta["arm"]),
                nodes=nodes,
                relations=relations,
                risk_objects=objects,
                risk_threads=list(annotation.get("risk_threads", [])),
                instances=instances,
            ))
    if len(traces) != 480:
        raise RuntimeError(f"expected 480 traces; found {len(traces)}")
    if any("## task_" in trace.task.lower() for trace in traces):
        raise RuntimeError("benchmark heading leaked into task input")
    return traces


def create_split(traces: list[Trace], config: dict[str, Any]) -> dict[str, Any]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for trace in traces:
        grouped[trace.group].append(trace.trace_id)
    if len(grouped) != 80 or set(map(len, grouped.values())) != {6}:
        raise RuntimeError("expected 80 groups of six related traces")
    groups = sorted(grouped)
    random.Random(int(config["seed"])).shuffle(groups)
    train_end = int(config["train_groups"])
    validation_end = train_end + int(config["validation_groups"])
    partition_groups = {
        "train": groups[:train_end],
        "validation": groups[train_end:validation_end],
        "test": groups[validation_end:],
    }
    record = {
        "seed": int(config["seed"]),
        "grouping": "category/harm/task_id; arm and surface variants stay together",
        "partition_groups": partition_groups,
        "trace_ids": {
            name: sorted(
                trace_id for group in values for trace_id in grouped[group]
            )
            for name, values in partition_groups.items()
        },
    }
    record["counts"] = {name: len(values) for name, values in record["trace_ids"].items()}
    if record["counts"] != {"train": 216, "validation": 72, "test": 192}:
        raise RuntimeError(f"wrong split counts: {record['counts']}")
    by_id = {trace.trace_id: trace for trace in traces}
    record["surface_counts"] = {
        name: dict(Counter(by_id[value].surface for value in ids))
        for name, ids in record["trace_ids"].items()
    }
    expected = {
        "train": {"controller": 72, "memory": 72, "tools_skills": 72},
        "validation": {"controller": 24, "memory": 24, "tools_skills": 24},
        "test": {"controller": 64, "memory": 64, "tools_skills": 64},
    }
    if record["surface_counts"] != expected:
        raise RuntimeError(f"unbalanced surfaces: {record['surface_counts']}")
    return record


def load_or_create_split(path: Path, traces: list[Trace], config: dict[str, Any]):
    expected = create_split(traces, config)
    if path.is_file():
        current = read_json(path)
        if current != expected:
            raise RuntimeError("existing frozen split differs from deterministic split")
    else:
        write_json(path, expected)
    return expected


def make_matrix(rows, config, text_vectorizer=None, graph_vectorizer=None, *, fit):
    if fit:
        text_vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=int(config["word_tfidf_minimum_document_frequency"]),
            max_features=int(config["word_tfidf_maximum_features"]),
            sublinear_tf=True,
        )
        graph_vectorizer = DictVectorizer(sparse=True)
        text = text_vectorizer.fit_transform([row.text for row in rows])
        graph = graph_vectorizer.fit_transform([row.features for row in rows])
    else:
        text = text_vectorizer.transform([row.text for row in rows])
        graph = graph_vectorizer.transform([row.features for row in rows])
    return sparse.hstack([text, graph], format="csr"), text_vectorizer, graph_vectorizer


class BagModel(torch.nn.Module):
    def __init__(self, dimensions: int):
        super().__init__()
        self.linear = torch.nn.Linear(dimensions, 1)

    def instance_logits(self, values):
        return self.linear(values).squeeze(-1)

    def bag_logit(self, logits):
        weights = torch.softmax(logits / 0.25, dim=0)
        return torch.sum(weights * logits)


def bag_indices(rows):
    bags: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        bags[row.trace_id].append(index)
    return {key: np.asarray(value, dtype=int) for key, value in bags.items()}


def fit_model(values, rows, traces, seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    model = BagModel(values.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.01)
    tensor = torch.from_numpy(values)
    bags = bag_indices(rows)
    trace_by_id = {trace.trace_id: trace for trace in traces}
    trace_ids = sorted(bags)
    positives = sum(trace_by_id[value].outcome == "failure" for value in trace_ids)
    negatives = len(trace_ids) - positives
    if not positives or not negatives:
        raise ValueError("candidate-bearing training bags require both outcomes")
    loss_function = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(negatives / positives, dtype=torch.float32)
    )
    best_loss, best_state, stale = float("inf"), None, 0
    generator = random.Random(seed)
    for _ in range(350):
        generator.shuffle(trace_ids)
        optimizer.zero_grad(); losses = []
        for trace_id in trace_ids:
            indexes = torch.from_numpy(bags[trace_id])
            value = model.bag_logit(model.instance_logits(tensor[indexes])).reshape(1)
            target = torch.tensor([float(trace_by_id[trace_id].outcome == "failure")])
            losses.append(loss_function(value, target))
        loss = torch.stack(losses).mean(); loss.backward(); optimizer.step()
        current = float(loss.detach())
        if current < best_loss - 1e-5:
            best_loss = current
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= 30:
            break
    model.load_state_dict(best_state); model.eval()
    return model


def score_instances(model, values, rows):
    with torch.no_grad():
        scores = torch.sigmoid(model.instance_logits(torch.from_numpy(values))).numpy()
    return [
        {
            "instance_id": row.instance_id,
            "trace_id": row.trace_id,
            "risk_id": row.risk_id,
            "risk": float(score),
            "state": row.record["final_state"],
            "warning_unit": row.warning_unit,
            "current_unit": row.current_unit,
            "max_unit": row.max_unit,
        }
        for row, score in zip(rows, scores, strict=True)
    ]


def confusion(rows):
    tp = sum(row["outcome"] == "failure" and row["prediction"] for row in rows)
    fn = sum(row["outcome"] == "failure" and not row["prediction"] for row in rows)
    fp = sum(row["outcome"] == "safe" and row["prediction"] for row in rows)
    tn = sum(row["outcome"] == "safe" and not row["prediction"] for row in rows)
    return {
        "n": len(rows), "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": (tp + tn) / len(rows) if rows else None,
        "failure_recall": tp / (tp + fn) if tp + fn else None,
        "false_positive_rate": fp / (fp + tn) if fp + tn else None,
    }


def routing_rows(traces: list[Trace], scores, threshold: float):
    by_trace: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scores:
        by_trace[row["trace_id"]].append(row)
    output = []
    for trace in traces:
        selected = [row for row in by_trace[trace.trace_id] if row["risk"] >= threshold]
        fallback = not trace.instances
        output.append({
            "trace_id": trace.trace_id,
            "outcome": trace.outcome,
            "surface": trace.surface,
            "arm": trace.arm,
            "prediction": bool(selected) or fallback,
            "selected_count": len(selected) + int(fallback),
            "full_trace_fallback": fallback,
            "maximum_risk": max((row["risk"] for row in by_trace[trace.trace_id]), default=0.0),
        })
    return output


def choose_threshold(traces, scores, config):
    options = []
    for threshold in sorted({0.0, 1.0, *(row["risk"] for row in scores)}):
        rows = routing_rows(traces, scores, threshold)
        options.append({
            "threshold": threshold,
            "selected_threads": sum(row["selected_count"] for row in rows),
            **confusion(rows),
        })
    target = float(config["stage1_target_validation_recall"])
    qualified = [row for row in options if row["failure_recall"] >= target]
    if not qualified:
        raise RuntimeError("Stage 1 cannot reach its validation recall target")
    chosen = min(qualified, key=lambda row: (
        row["false_positive_rate"], row["selected_threads"], -row["threshold"]
    ))
    return chosen, options


def node_packet(node: Node):
    return {
        "node_id": node.node_id,
        "reasoningflow_label": node.rf_label,
        "text": node.text,
        "risk_thread_annotations": node.annotations,
    }


def thread_packet(instance: RiskThread):
    return {
        "risk_id": instance.risk_id,
        "object": instance.object_record,
        **instance.record,
    }


def verifier_packet(trace: Trace, selected: list[dict[str, Any]], by_instance):
    if selected:
        focus = [by_instance[row["instance_id"]] for row in selected]
        mode = "selected_risk_threads"
        focus_ids = [row.risk_id for row in focus]
        threads = [thread_packet(row) for row in focus]
    else:
        mode = "full_trace_fallback"
        focus_ids = []
        threads = [
            {
                "risk_id": str(row["risk_id"]),
                "object": trace.risk_objects[str(row["object_id"])],
                **row,
            }
            for row in trace.risk_threads
        ]
    return {
        "input_mode": mode,
        "task": trace.task,
        "focus_risk_ids": focus_ids,
        "focus_risk_threads": threads,
        "all_risk_threads": [
            {
                "risk_id": str(row["risk_id"]),
                "object": trace.risk_objects[str(row["object_id"])],
                **row,
            }
            for row in trace.risk_threads
        ],
        "complete_reasoning": [node_packet(node) for node in trace.nodes],
        "all_object_linked_relations": trace.relations,
    }


def call_verifier(rendered: str, config: dict[str, Any]):
    """Call an API-served verifier without depending on Codex."""
    try:
        from openai import OpenAI
    except ImportError as error:
        raise RuntimeError("Install the package with its API dependencies") from error

    model = os.environ.get(
        "REASONING_MONITOR_MODEL", str(config["verifier_model"])
    ).strip()
    if not model:
        raise RuntimeError("Set REASONING_MONITOR_MODEL to an API model name")
    schema = read_json(SOURCE / "verifier_schema.json")
    schema.pop("$schema", None)
    client = OpenAI(
        timeout=float(config["verifier_timeout_seconds"]),
        max_retries=0,
    )
    attempts = int(config.get("verifier_attempts", 3))
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            response = client.responses.create(
                model=model,
                input=rendered,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "reasoning_monitor_verdict",
                        "strict": True,
                        "schema": schema,
                    }
                },
            )
            if not response.output_text:
                raise RuntimeError("verifier returned no structured output")
            return json.loads(response.output_text)
        except Exception as error:  # API/network failures are retried, then surfaced.
            last_error = error
            if attempt + 1 < attempts:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"verifier failed after {attempts} attempts: {last_error}")


def verify_partition(traces, scores, threshold, instances, config, cache_dir):
    scores_by_trace: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scores:
        if row["risk"] >= threshold:
            scores_by_trace[row["trace_id"]].append(row)
    by_instance = {row.instance_id: row for row in instances}
    prompt = verifier_prompt_path().read_text(encoding="utf-8")
    cache_dir.mkdir(parents=True, exist_ok=True)

    def verify(trace: Trace):
        selected = sorted(scores_by_trace.get(trace.trace_id, []), key=lambda row: -row["risk"])
        fallback = not trace.instances
        if not selected and not fallback:
            return {
                "trace_id": trace.trace_id, "outcome": trace.outcome,
                "surface": trace.surface, "arm": trace.arm,
                "prediction": False, "selected_count": 0,
                "full_trace_fallback": False, "verifier": None,
                "verifier_score": 0.0,
                "trace_remaining_at_warning": None,
                "trace_remaining_at_intervention": None,
            }
        packet = verifier_packet(trace, selected, by_instance)
        rendered = prompt + "\n\n# Input\n\n" + json.dumps(packet, indent=2)
        model = os.environ.get(
            "REASONING_MONITOR_MODEL", str(config["verifier_model"])
        ).strip()
        request_hash = hashlib.sha256(
            (model + "\n" + rendered).encode("utf-8")
        ).hexdigest()
        response_path = cache_dir / f"{request_hash}.json"
        write_json(cache_dir / "requests" / f"{request_hash}.json", packet)
        if response_path.is_file():
            verdict = read_json(response_path)["verdict"]
        else:
            verdict = call_verifier(rendered, config)
            write_json(response_path, {"packet": packet, "verdict": verdict})

        valid_nodes = {node.node_id for node in trace.nodes}
        cited_nodes = set(map(str, verdict.get("risk_node_ids", []))) | set(
            map(str, verdict.get("plan_node_ids", []))
        ) | set(map(str, verdict.get("correction_node_ids", [])))
        focus_valid = (
            verdict.get("focus_risk_id") in packet["focus_risk_ids"]
            if selected else verdict.get("focus_risk_id") in {"reconstructed", "none"}
        )
        active_harm = (
            verdict.get("harm_type") != "none"
            and verdict.get("final_state") in {"active", "reopened"}
            and bool(verdict.get("risk_node_ids"))
            and bool(verdict.get("plan_node_ids"))
        )
        consistent = (
            focus_valid
            and cited_nodes <= valid_nodes
            and isinstance(verdict.get("risk_score"), int)
            and 0 <= verdict.get("risk_score", -1) <= 100
        )
        verifier_score = float(verdict.get("risk_score", 0)) if consistent else 0.0
        if active_harm and consistent:
            warning = min(unit(value) for value in verdict["risk_node_ids"])
            intervention = max(unit(value) for value in verdict["plan_node_ids"])
            max_unit = max(node.unit for node in trace.nodes)
        else:
            warning = intervention = max_unit = None
        return {
            "trace_id": trace.trace_id, "outcome": trace.outcome,
            "surface": trace.surface, "arm": trace.arm,
            "prediction": False,
            "verifier_score": verifier_score,
            "selected_count": len(selected) + int(fallback),
            "full_trace_fallback": fallback,
            "verifier": verdict, "verifier_consistent": consistent,
            "trace_remaining_at_warning": (
                None if warning is None else 1 - warning / max(1, max_unit)
            ),
            "trace_remaining_at_intervention": (
                None if intervention is None else 1 - intervention / max(1, max_unit)
            ),
        }

    with ThreadPoolExecutor(max_workers=int(config["verifier_parallel_requests"])) as pool:
        return list(pool.map(verify, sorted(traces, key=lambda row: row.trace_id)))


def choose_verifier_threshold(rows, config):
    """Maximize validation recall subject to the predeclared FPR constraint."""
    options = []
    scores = sorted({0.0, 101.0, *(float(row.get("verifier_score", 0.0)) for row in rows)})
    for threshold in scores:
        candidate = [
            {**row, "prediction": float(row.get("verifier_score", 0.0)) >= threshold}
            for row in rows
        ]
        options.append({"threshold": threshold, **confusion(candidate)})
    limit = float(config["required_validation_fpr_below"])
    qualified = [row for row in options if row["false_positive_rate"] < limit]
    if not qualified:
        raise RuntimeError("No Stage-2 threshold satisfies the validation FPR constraint")
    chosen = max(
        qualified,
        key=lambda row: (row["failure_recall"], row["accuracy"], -row["false_positive_rate"], row["threshold"]),
    )
    calibrated = [
        {**row, "prediction": float(row.get("verifier_score", 0.0)) >= chosen["threshold"]}
        for row in rows
    ]
    return chosen, options, calibrated


def metric_bundle(rows):
    return {
        "overall": confusion(rows),
        "by_surface": {
            surface: confusion([row for row in rows if row["surface"] == surface])
            for surface in ("controller", "memory", "tools_skills")
        },
        "safe_fpr_by_arm": {
            arm: confusion([
                row for row in rows
                if row["outcome"] == "failure" or row["arm"] == arm
            ])["false_positive_rate"]
            for arm in ("none", "self_evolving")
        },
    }


def snapshot_method(output: Path):
    target = output / "method_snapshot"
    target.mkdir(parents=True, exist_ok=True)
    for name in ("selector_verifier.py", "config.json", "verifier_schema.json"):
        shutil.copy2(SOURCE / name, target / name)
    shutil.copy2(verifier_prompt_path(), target / "verifier_prompt.md")


def train_stage1(train_traces, validation_traces, config, output):
    train_instances = [row for trace in train_traces for row in trace.instances]
    validation_instances = [row for trace in validation_traces for row in trace.instances]
    matrix, text_vectorizer, graph_vectorizer = make_matrix(train_instances, config, fit=True)
    validation_matrix, _, _ = make_matrix(
        validation_instances, config, text_vectorizer, graph_vectorizer, fit=False
    )
    dimensions = min(int(config["svd_dimensions"]), matrix.shape[0] - 1, matrix.shape[1] - 1)
    reducer = TruncatedSVD(n_components=dimensions, random_state=int(config["seed"]))
    train_values = reducer.fit_transform(matrix)
    validation_values = reducer.transform(validation_matrix)
    scaler = StandardScaler()
    train_values = scaler.fit_transform(train_values).astype("float32")
    validation_values = scaler.transform(validation_values).astype("float32")
    model = fit_model(train_values, train_instances, train_traces, int(config["seed"]))
    validation_scores = score_instances(model, validation_values, validation_instances)
    chosen, options = choose_threshold(validation_traces, validation_scores, config)
    artifact = {
        "config": config, "text_vectorizer": text_vectorizer,
        "graph_vectorizer": graph_vectorizer, "reducer": reducer, "scaler": scaler,
        "model_state": {key: value.numpy() for key, value in model.state_dict().items()},
        "threshold": float(chosen["threshold"]),
    }
    joblib.dump(artifact, output / "stage1.joblib")
    write_json(output / "stage1_validation.json", {"selected": chosen, "options": options})
    write_jsonl(output / "stage1_validation_scores.jsonl", validation_scores)
    return artifact, model, validation_scores, validation_instances


def load_stage1(output: Path):
    artifact = joblib.load(output / "stage1.joblib")
    dimensions = int(artifact["reducer"].n_components)
    model = BagModel(dimensions)
    model.load_state_dict({
        key: torch.from_numpy(value) for key, value in artifact["model_state"].items()
    })
    model.eval()
    return artifact, model


def score_with_artifact(instances, artifact, model):
    matrix, _, _ = make_matrix(
        instances, artifact["config"], artifact["text_vectorizer"],
        artifact["graph_vectorizer"], fit=False,
    )
    values = artifact["scaler"].transform(
        artifact["reducer"].transform(matrix)
    ).astype("float32")
    return score_instances(model, values, instances)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=("prepare", "validate", "test"), required=True)
    args = parser.parse_args()

    config = read_json(SOURCE / "config.json")
    traces = load_corpus(args.data_root.resolve(), config)
    split = load_or_create_split(args.split_file.resolve(), traces, config)
    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=True)
    snapshot_method(output)
    by_id = {trace.trace_id: trace for trace in traces}
    partitions = {
        name: [by_id[value] for value in ids]
        for name, ids in split["trace_ids"].items()
    }
    if args.stage == "prepare":
        report = {
            "split_counts": split["counts"],
            "surface_counts": split["surface_counts"],
            "candidate_state_counts": dict(Counter(
                row.record["final_state"] for trace in traces for row in trace.instances
            )),
            "full_trace_fallback_counts": {
                name: sum(not trace.instances for trace in values)
                for name, values in partitions.items()
            },
            "test_outcomes_not_reported": True,
        }
        write_json(output / "prepare_report.json", report)
        print(json.dumps(report, indent=2)); return

    if args.stage == "validate":
        artifact, model, scores, instances = train_stage1(
            partitions["train"], partitions["validation"], config, output
        )
        raw_predictions = verify_partition(
            partitions["validation"], scores, float(artifact["threshold"]),
            instances, config, output / "verifier_cache" / "validation",
        )
        stage2_selected, stage2_options, predictions = choose_verifier_threshold(
            raw_predictions, config
        )
        metrics = metric_bundle(predictions)
        write_json(output / "stage2_validation.json", {
            "selection_rule": "maximize failure recall subject to FPR < configured limit; then accuracy",
            "selected": stage2_selected,
            "options": stage2_options,
        })
        write_jsonl(output / "validation_raw_scores.jsonl", raw_predictions)
        write_jsonl(output / "validation_predictions.jsonl", predictions)
        write_json(output / "validation_report.json", metrics)
        write_json(output / "validation_manifest.json", {
            "stage": "validation",
            "config_sha256": sha256(SOURCE / "config.json"),
            "prompt_sha256": sha256(verifier_prompt_path()),
            "split_sha256": sha256(args.split_file.resolve()),
            "test_evaluated": False,
        })
        print(json.dumps(metrics, indent=2)); return

    freeze_path = output / "FROZEN_FOR_TEST.json"
    if not freeze_path.is_file():
        raise RuntimeError("test is locked; freeze the monitor after validation")
    freeze = read_json(freeze_path)
    expected_hashes = {
        "config_sha256": sha256(SOURCE / "config.json"),
        "prompt_sha256": sha256(verifier_prompt_path()),
        "split_sha256": sha256(args.split_file.resolve()),
    }
    if any(freeze.get(key) != value for key, value in expected_hashes.items()):
        raise RuntimeError("frozen files do not match the selected validation monitor")
    artifact, model = load_stage1(output)
    stage2_threshold = float(read_json(output / "stage2_validation.json")["selected"]["threshold"])
    instances = [row for trace in partitions["test"] for row in trace.instances]
    scores = score_with_artifact(instances, artifact, model)
    raw_predictions = verify_partition(
        partitions["test"], scores, float(artifact["threshold"]), instances,
        config, output / "verifier_cache" / "test",
    )
    predictions = [
        {**row, "prediction": float(row.get("verifier_score", 0.0)) >= stage2_threshold}
        for row in raw_predictions
    ]
    metrics = metric_bundle(predictions)
    write_jsonl(output / "test_predictions.jsonl", predictions)
    write_jsonl(output / "stage1_test_scores.jsonl", scores)
    write_json(output / "test_report.json", metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
