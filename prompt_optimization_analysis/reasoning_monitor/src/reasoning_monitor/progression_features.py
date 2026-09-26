#!/usr/bin/env python3
"""Extract progression features and select thresholds using training data only."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from scipy import sparse
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.svm import LinearSVC


SURFACES = ("controller", "memory", "tools_skills")


def load_base(path: Path):
    spec = importlib.util.spec_from_file_location("risk_thread_base", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def trace_text(trace) -> str:
    nodes = "\n".join(
        f"[{node.node_id} RF={node.rf_label}] {node.text}" for node in trace.nodes
    )
    threads = "\n".join(instance.text for instance in trace.instances)
    return f"TASK:\n{trace.task}\n\nREASONING:\n{nodes}\n\nRISK THREADS:\n{threads}"


def trace_features(trace) -> dict[str, float]:
    features: dict[str, float] = {"bias": 1.0}

    def add(key: str, value: float = 1.0):
        features[key] = features.get(key, 0.0) + value

    for node in trace.nodes:
        add(f"rf_count={node.rf_label}")
        add(f"rf_present={node.rf_label}", 0.0)
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

    # Preserve object-linked progressions instead of
    # reducing them to isolated labels.
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
        yy = y[indexes]; pp = prediction[indexes]
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
            value: one(np.flatnonzero(surfaces == value)) for value in SURFACES
        },
    }


def select_threshold(y, scores, surfaces, fpr_limit=0.10):
    candidates = []
    for threshold in sorted({float("inf"), float("-inf"), *map(float, scores)}):
        result = metrics(y, scores >= threshold, surfaces)
        overall = result["overall"]
        if overall["false_positive_rate"] < fpr_limit:
            surface_recalls = [
                result["by_surface"][surface]["failure_recall"] for surface in SURFACES
            ]
            candidates.append((
                min(surface_recalls), overall["failure_recall"], overall["accuracy"],
                -overall["false_positive_rate"], threshold, result,
            ))
    if not candidates:
        raise RuntimeError("no threshold satisfies the FPR constraint")
    return max(candidates)


def vectorize(train_rows, test_rows):
    word = TfidfVectorizer(
        ngram_range=(1, 2), min_df=2, max_df=0.98, max_features=30000,
        sublinear_tf=True, strip_accents="unicode",
    )
    char = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=30000,
        sublinear_tf=True,
    )
    graph = DictVectorizer(sparse=True)
    train_text = [trace_text(row) for row in train_rows]
    test_text = [trace_text(row) for row in test_rows]
    x_train = sparse.hstack([
        word.fit_transform(train_text), char.fit_transform(train_text),
        graph.fit_transform([trace_features(row) for row in train_rows]),
    ], format="csr")
    x_test = sparse.hstack([
        word.transform(test_text), char.transform(test_text),
        graph.transform([trace_features(row) for row in test_rows]),
    ], format="csr")
    return x_train, x_test, (word, char, graph)


def make_model(name: str, parameter: float):
    if name == "linear_svc":
        return LinearSVC(C=parameter, class_weight="balanced", max_iter=20000)
    if name == "logistic":
        return LogisticRegression(
            C=parameter, class_weight="balanced", solver="liblinear", max_iter=5000,
        )
    if name == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=800, max_depth=int(parameter), min_samples_leaf=3,
            class_weight="balanced", max_features="sqrt", random_state=20260919,
            n_jobs=-1,
        )
    raise ValueError(name)


def model_scores(model, values):
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(values), dtype=float)
    return np.asarray(model.predict_proba(values)[:, 1], dtype=float)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-monitor", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=("select", "validate", "test"), required=True)
    args = parser.parse_args()

    base = load_base(args.base_monitor.resolve())
    config = base.read_json(args.base_monitor.resolve().parent / "config.json")
    traces = base.load_corpus(args.data_root.resolve(), config)
    split = base.read_json(args.split_file.resolve())
    by_id = {trace.trace_id: trace for trace in traces}
    partitions = {
        name: [by_id[value] for value in ids] for name, ids in split["trace_ids"].items()
    }
    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=True)

    if args.stage == "select":
        rows = partitions["train"]
        y = np.array([row.outcome == "failure" for row in rows], dtype=int)
        groups = np.array([row.group for row in rows])
        surfaces = np.array([row.surface for row in rows])
        folds = list(StratifiedGroupKFold(
            n_splits=6, shuffle=True, random_state=20260919
        ).split(rows, y, groups))
        specifications = [
            *(('linear_svc', value) for value in (0.01, 0.03, 0.1, 0.3, 1.0, 3.0)),
            *(('logistic', value) for value in (0.03, 0.1, 0.3, 1.0, 3.0, 10.0)),
            *(('extra_trees', value) for value in (3, 5, 8, 12)),
        ]
        all_results = []
        for name, parameter in specifications:
            oof = np.zeros(len(rows), dtype=float)
            for train_index, held_index in folds:
                train_rows = [rows[index] for index in train_index]
                held_rows = [rows[index] for index in held_index]
                if name == "extra_trees":
                    vectorizer = DictVectorizer(sparse=False)
                    x_train = vectorizer.fit_transform([trace_features(row) for row in train_rows])
                    x_held = vectorizer.transform([trace_features(row) for row in held_rows])
                else:
                    x_train, x_held, _ = vectorize(train_rows, held_rows)
                model = make_model(name, parameter)
                model.fit(x_train, y[train_index])
                oof[held_index] = model_scores(model, x_held)
            selected = select_threshold(y, oof, surfaces)
            minimum_surface_recall, _, _, _, threshold, report = selected
            all_results.append({
                "model": name, "parameter": parameter, "threshold": threshold,
                "minimum_surface_recall": minimum_surface_recall,
                "metrics": report,
            })
        best = max(all_results, key=lambda row: (
            row["minimum_surface_recall"],
            row["metrics"]["overall"]["failure_recall"],
            row["metrics"]["overall"]["accuracy"],
            -row["metrics"]["overall"]["false_positive_rate"],
        ))
        (output / "train_cv_selection.json").write_text(json.dumps({
            "selection_data": "train only", "selected": best, "all_results": all_results,
        }, indent=2) + "\n")
        print(json.dumps(best, indent=2)); return

    selection = json.loads((output / "train_cv_selection.json").read_text())["selected"]
    train_rows = partitions["train"]
    target_rows = partitions["validation"] if args.stage == "validate" else partitions["test"]
    y_train = np.array([row.outcome == "failure" for row in train_rows], dtype=int)
    if selection["model"] == "extra_trees":
        vectorizer = DictVectorizer(sparse=False)
        x_train = vectorizer.fit_transform([trace_features(row) for row in train_rows])
        x_target = vectorizer.transform([trace_features(row) for row in target_rows])
        vectorizers = (vectorizer,)
    else:
        x_train, x_target, vectorizers = vectorize(train_rows, target_rows)
    model = make_model(selection["model"], float(selection["parameter"]))
    model.fit(x_train, y_train)
    scores = model_scores(model, x_target)
    threshold = float(selection["threshold"])
    prediction = scores >= threshold
    outcomes = np.array([row.outcome == "failure" for row in target_rows], dtype=int)
    surfaces = np.array([row.surface for row in target_rows])
    report = metrics(outcomes, prediction, surfaces)
    records = [
        {
            "trace_id": row.trace_id, "surface": row.surface,
            "outcome": row.outcome, "score": float(score), "prediction": bool(pred),
        }
        for row, score, pred in zip(target_rows, scores, prediction, strict=True)
    ]
    suffix = "validation" if args.stage == "validate" else "test"
    (output / f"{suffix}_predictions.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in records)
    )
    (output / f"{suffix}_report.json").write_text(json.dumps(report, indent=2) + "\n")
    joblib.dump({"model": model, "vectorizers": vectorizers, "selection": selection}, output / "monitor.joblib")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
