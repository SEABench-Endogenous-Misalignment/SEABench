#!/usr/bin/env python3
"""ExtraTrees progression scorer with all harm-router features removed."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import StratifiedGroupKFold


def import_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def make_model(depth, leaf, feature_fraction):
    return ExtraTreesClassifier(
        n_estimators=300,
        max_depth=depth,
        min_samples_leaf=leaf,
        max_features=feature_fraction,
        class_weight="balanced",
        random_state=20260926,
        n_jobs=-1,
    )


def records(rows, scores, threshold):
    return [
        {
            "trace_id": row.trace_id,
            "surface": row.surface,
            "outcome": row.outcome,
            "score": float(score),
            "prediction": bool(score >= threshold),
        }
        for row, score in zip(rows, scores, strict=True)
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-monitor", type=Path, required=True)
    parser.add_argument("--feature-module", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=("select", "validate", "test"), required=True)
    args = parser.parse_args()

    base = import_file("no_harm_base", args.base_monitor.resolve())
    features = import_file("no_harm_features", args.feature_module.resolve())
    config = base.read_json(args.base_monitor.resolve().parent / "config.json")
    traces = base.load_corpus(args.data_root.resolve(), config)
    split = base.read_json(args.split_file.resolve())
    by_id = {trace.trace_id: trace for trace in traces}
    partitions = {
        name: [by_id[trace_id] for trace_id in ids]
        for name, ids in split["trace_ids"].items()
    }
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    if args.stage == "select":
        rows = partitions["train"]
        outcomes = np.asarray([row.outcome == "failure" for row in rows], dtype=int)
        groups = np.asarray([row.group for row in rows])
        surfaces = np.asarray([row.surface for row in rows])
        folds = list(StratifiedGroupKFold(
            n_splits=6, shuffle=True, random_state=20260926
        ).split(rows, outcomes, groups))
        specifications = [(16, 2, 0.30)]
        results = []
        for depth, leaf, fraction in specifications:
            out_of_fold = np.zeros(len(rows), dtype=float)
            for fit_indexes, held_indexes in folds:
                fit_rows = [rows[index] for index in fit_indexes]
                held_rows = [rows[index] for index in held_indexes]
                vectorizer = DictVectorizer(sparse=False)
                fit_values = vectorizer.fit_transform([
                    features.trace_features(row) for row in fit_rows
                ])
                held_values = vectorizer.transform([
                    features.trace_features(row) for row in held_rows
                ])
                model = make_model(depth, leaf, fraction)
                model.fit(fit_values, outcomes[fit_indexes])
                out_of_fold[held_indexes] = model.predict_proba(held_values)[:, 1]
            selected = features.select_threshold(outcomes, out_of_fold, surfaces)
            minimum_surface_recall, _, _, _, threshold, report = selected
            results.append({
                "depth": depth,
                "leaf": leaf,
                "feature_fraction": fraction,
                "threshold": threshold,
                "minimum_surface_recall": minimum_surface_recall,
                "metrics": report,
            })
        chosen = max(results, key=lambda row: (
            row["minimum_surface_recall"],
            row["metrics"]["overall"]["failure_recall"],
            row["metrics"]["overall"]["accuracy"],
            -row["metrics"]["overall"]["false_positive_rate"],
        ))
        payload = {
            "selection_data": "training partition only",
            "ablation": "same ExtraTrees search with all harm-router features removed",
            "selected": chosen,
            "all_models": results,
        }
        (output / "train_cv_selection.json").write_text(
            json.dumps(payload, indent=2) + "\n"
        )
        print(json.dumps(chosen, indent=2))
        return

    if args.stage == "test":
        artifact = joblib.load(output / "monitor.joblib")
        target = partitions["test"]
        values = artifact["vectorizer"].transform([
            features.trace_features(row) for row in target
        ])
        scores = artifact["model"].predict_proba(values)[:, 1]
        threshold = float(artifact["selection"]["selected"]["threshold"])
        predictions = records(target, scores, threshold)
        report = features.metrics(
            np.asarray([row.outcome == "failure" for row in target], dtype=int),
            scores >= threshold,
            np.asarray([row.surface for row in target]),
        )
        (output / "test_predictions.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in predictions)
        )
        (output / "test_report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        return

    selection = json.loads((output / "train_cv_selection.json").read_text())
    train = partitions["train"]
    target = partitions["validation"]
    vectorizer = DictVectorizer(sparse=False)
    train_values = vectorizer.fit_transform([
        features.trace_features(row) for row in train
    ])
    target_values = vectorizer.transform([
        features.trace_features(row) for row in target
    ])
    chosen = selection["selected"]
    model = make_model(
        int(chosen["depth"]), int(chosen["leaf"]), chosen["feature_fraction"]
    )
    model.fit(
        train_values,
        np.asarray([row.outcome == "failure" for row in train], dtype=int),
    )
    scores = model.predict_proba(target_values)[:, 1]
    threshold = float(chosen["threshold"])
    predictions = records(target, scores, threshold)
    report = features.metrics(
        np.asarray([row.outcome == "failure" for row in target], dtype=int),
        scores >= threshold,
        np.asarray([row.surface for row in target]),
    )
    (output / "validation_predictions.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in predictions)
    )
    (output / "validation_report.json").write_text(json.dumps(report, indent=2) + "\n")
    joblib.dump({
        "vectorizer": vectorizer,
        "model": model,
        "selection": selection,
    }, output / "monitor.joblib")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
