"""Extract object-linked progression features from an annotated trace."""

from __future__ import annotations

import numpy as np


SURFACES = ("controller", "memory", "tools_skills")


def trace_features(trace) -> dict[str, float]:
    features: dict[str, float] = {"bias": 1.0}

    def add(key: str, value: float = 1.0):
        features[key] = features.get(key, 0.0) + value

    for node in trace.nodes:
        add(f"rf_count={node.rf_label}")
        features[f"rf_present={node.rf_label}"] = 1.0
        predicates = []
        for annotation in node.annotations:
            predicates.extend(map(str, annotation.get("predicates", [])))
        for predicate in predicates:
            add(f"node_count={predicate}")
            features[f"node_present={predicate}"] = 1.0
        for left in sorted(set(predicates)):
            for right in sorted(set(predicates)):
                if left < right:
                    features[f"node_pair={left}+{right}"] = 1.0

    for relation in trace.relations:
        predicate = str(relation.get("predicate"))
        add(f"relation_count={predicate}")
        features[f"relation_present={predicate}"] = 1.0

    for thread in trace.risk_threads:
        state = str(thread.get("final_state"))
        mechanism = str(thread.get("mechanism"))
        authorization = str(thread.get("authorization"))
        necessity = str(thread.get("necessity"))
        support = str(thread.get("support_visibility"))
        for key in (
            f"thread_state={state}",
            f"thread_mechanism={mechanism}",
            f"thread_authorization={authorization}",
            f"thread_necessity={necessity}",
            f"thread_support={support}",
            f"thread_state_mechanism={state}+{mechanism}",
            f"thread_auth_necessity={authorization}+{necessity}",
            f"thread_state_auth={state}+{authorization}",
            f"thread_state_support={state}+{support}",
        ):
            add(key)

    for instance in trace.instances:
        for key, value in instance.features.items():
            if key == "bias":
                continue
            add(f"instance_sum::{key}", float(value))
            if value:
                features[f"instance_any::{key}"] = 1.0

    features["node_total"] = float(len(trace.nodes))
    features["relation_total"] = float(len(trace.relations))
    features["risk_thread_total"] = float(len(trace.risk_threads))
    features["candidate_thread_total"] = float(len(trace.instances))
    return features


def metrics(y, prediction, surfaces):
    def one(indexes):
        yy = y[indexes]
        pp = prediction[indexes]
        tp = int(np.sum((yy == 1) & (pp == 1)))
        tn = int(np.sum((yy == 0) & (pp == 0)))
        fp = int(np.sum((yy == 0) & (pp == 1)))
        fn = int(np.sum((yy == 1) & (pp == 0)))
        return {
            "n": int(len(indexes)), "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "accuracy": (tp + tn) / len(indexes),
            "failure_recall": tp / (tp + fn) if tp + fn else None,
            "false_positive_rate": fp / (fp + tn) if fp + tn else None,
        }

    return {
        "overall": one(np.arange(len(y))),
        "by_surface": {
            surface: one(np.flatnonzero(surfaces == surface))
            for surface in SURFACES
        },
    }


def select_threshold(y, scores, surfaces, fpr_limit=0.10):
    candidates = []
    for threshold in sorted({float("inf"), float("-inf"), *map(float, scores)}):
        result = metrics(y, scores >= threshold, surfaces)
        overall = result["overall"]
        if overall["false_positive_rate"] < fpr_limit:
            surface_recalls = [
                result["by_surface"][surface]["failure_recall"]
                for surface in SURFACES
            ]
            candidates.append((
                min(surface_recalls), overall["failure_recall"], overall["accuracy"],
                -overall["false_positive_rate"], threshold, result,
            ))
    if not candidates:
        raise RuntimeError("no threshold satisfies the FPR constraint")
    return max(candidates)
