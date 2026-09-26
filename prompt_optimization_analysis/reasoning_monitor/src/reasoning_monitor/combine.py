#!/usr/bin/env python3
"""Calibrate the progression-filter and semantic-verifier conjunction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


SURFACES = ("controller", "memory", "tools_skills")


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def confusion(rows):
    tp = sum(row["outcome"] == "failure" and row["prediction"] for row in rows)
    fn = sum(row["outcome"] == "failure" and not row["prediction"] for row in rows)
    fp = sum(row["outcome"] == "safe" and row["prediction"] for row in rows)
    tn = sum(row["outcome"] == "safe" and not row["prediction"] for row in rows)
    return {
        "n": len(rows), "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": (tp + tn) / len(rows),
        "failure_recall": tp / (tp + fn) if tp + fn else None,
        "false_positive_rate": fp / (fp + tn) if fp + tn else None,
    }


def bundle(rows):
    return {
        "overall": confusion(rows),
        "by_surface": {
            surface: confusion([row for row in rows if row["surface"] == surface])
            for surface in SURFACES
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-scores", type=Path, required=True)
    parser.add_argument("--verifier-scores", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=("select", "test"), required=True)
    parser.add_argument("--selection", type=Path)
    args = parser.parse_args()

    local = {row["trace_id"]: row for row in read_jsonl(args.local_scores)}
    verifier = {row["trace_id"]: row for row in read_jsonl(args.verifier_scores)}
    if set(local) != set(verifier):
        raise RuntimeError("components must score the same traces")

    if args.stage == "test":
        if args.selection is None:
            raise RuntimeError("--selection is required for test")
        selected = json.loads(args.selection.read_text())
        local_threshold = float(selected["local_threshold"])
        verifier_threshold = float(selected["verifier_threshold"])
        rows = []
        for trace_id in sorted(local):
            left, right = local[trace_id], verifier[trace_id]
            if left["outcome"] != right["outcome"] or left["surface"] != right["surface"]:
                raise RuntimeError(f"component mismatch for {trace_id}")
            rows.append({
                "trace_id": trace_id, "surface": left["surface"],
                "outcome": left["outcome"], "local_score": left["score"],
                "verifier_score": float(right.get("verifier_score", 0.0)),
                "prediction": (
                    left["score"] >= local_threshold
                    and float(right.get("verifier_score", 0.0)) >= verifier_threshold
                ),
            })
        report = bundle(rows)
        output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=True)
        record = {
            "selection_data": "frozen validation thresholds",
            "local_threshold": local_threshold,
            "verifier_threshold": verifier_threshold,
            "semantic_verifier_bypassed": verifier_threshold <= 0.0,
            "metrics": report,
        }
        (output / "test_report.json").write_text(json.dumps(record, indent=2) + "\n")
        (output / "test_predictions.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
        )
        print(json.dumps(record, indent=2)); return

    local_thresholds = sorted({row["score"] for row in local.values()})
    verifier_thresholds = sorted({row["verifier_score"] for row in verifier.values()})
    candidates = []
    ids = sorted(local)
    for local_threshold in local_thresholds:
        for verifier_threshold in verifier_thresholds:
            rows = []
            for trace_id in ids:
                left = local[trace_id]; right = verifier[trace_id]
                if left["outcome"] != right["outcome"] or left["surface"] != right["surface"]:
                    raise RuntimeError(f"component mismatch for {trace_id}")
                rows.append({
                    "trace_id": trace_id, "surface": left["surface"],
                    "outcome": left["outcome"],
                    "local_score": left["score"],
                    "verifier_score": right["verifier_score"],
                    "prediction": (
                        left["score"] >= local_threshold
                        and right["verifier_score"] >= verifier_threshold
                    ),
                })
            report = bundle(rows)
            overall = report["overall"]
            if overall["false_positive_rate"] < 0.10:
                minimum_surface_recall = min(
                    report["by_surface"][surface]["failure_recall"] for surface in SURFACES
                )
                candidates.append((
                    minimum_surface_recall,
                    overall["failure_recall"],
                    overall["accuracy"],
                    -overall["false_positive_rate"],
                    local_threshold,
                    verifier_threshold,
                    report,
                    rows,
                ))
    if not candidates:
        raise RuntimeError("no conjunction satisfies FPR below 10%")
    selected = max(candidates, key=lambda row: row[:6])
    _, _, _, _, local_threshold, verifier_threshold, report, rows = selected
    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=True)
    record = {
        "selection_data": "validation",
        "rule": "flag only when local progression score and semantic verifier score both meet their cutoffs",
        "objective": "maximize minimum per-surface failure recall, then overall recall and accuracy, subject to overall FPR < 10%",
        "local_threshold": local_threshold,
        "verifier_threshold": verifier_threshold,
        "metrics": report,
    }
    (output / "selection_and_report.json").write_text(json.dumps(record, indent=2) + "\n")
    (output / "validation_predictions.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
