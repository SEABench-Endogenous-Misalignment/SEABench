#!/usr/bin/env python3
"""Run the original staged ReasoningFlow annotator over extracted SEABench traces."""
from __future__ import annotations

import argparse
import concurrent.futures
import os
import tempfile
from functools import partial
from pathlib import Path

import llm_labeler as rf
from repair_text_partitions import repair_document
from seabench_common import DEFAULT_OUT, DEFAULT_TRACES, convert_trace, iter_trace_files, read_json, write_json


def annotate_document(datum: dict, model: str, workers: int, node_retry_attempts: int = 3) -> dict:
    question = datum["raw_text"]["question"]
    response = datum["raw_text"]["response"]
    if not response.strip():
        raise ValueError("trace contains no generated reasoning")

    question_units, question_bounds = rf.tokenize_and_align(question, model)
    response_units, response_bounds = rf.tokenize_and_align(response, model)
    nodes = [
        rf._create_context_node(i, question_bounds[i], question_bounds[i + 1],
                                question[question_bounds[i]:question_bounds[i + 1]])
        for i in range(len(question_units))
    ]
    nodes.extend(
        rf._create_response_node(i, response_bounds[i], response_bounds[i + 1], None,
                                 response[response_bounds[i]:response_bounds[i + 1]])
        for i in range(len(response_units))
    )
    nodes = [node for node in nodes if node["text"].strip()]
    datum["nodes"] = nodes
    # The original pipeline drops whitespace-only alignment nodes. Absorb those
    # uncovered separators into adjacent semantic nodes so SEABench retains an
    # exact, lossless partition of its multi-paragraph traces.
    repair_document(datum)

    response_ids = {node["id"] for node in nodes if node["source"] == "response"}
    by_id: dict = {}
    missing: set = response_ids
    for attempt in range(1, node_retry_attempts + 1):
        labels = rf.node_classification(nodes, question, model)
        by_id = {item["node_id"]: item["label"] for item in labels["responses"]}
        missing = response_ids - by_id.keys()
        if not missing:
            break
        print(
            f"node classifier omitted {sorted(missing)}; retry {attempt}/{node_retry_attempts}",
            flush=True,
        )
    else:
        raise ValueError(
            f"node classifier omitted {sorted(missing)} after {node_retry_attempts} attempts"
        )
    for node in nodes:
        if node["source"] == "response":
            node["label"] = by_id[node["id"]]
    datum["nodes"] = nodes

    prompt = rf.PROMPTS["update_conclusion"].replace("<<input>>", rf.format_conclusion_prompt(datum))
    result = rf.call_llm(prompt, llm_model_name=model, schema=rf.ConclusionNodeList,
                         thinking_level="high")
    conclusion_ids = set(result["conclusion_node_ids"])
    response_ids = {n["id"] for n in nodes if n["source"] == "response"}
    unknown = conclusion_ids - response_ids
    if unknown:
        raise ValueError(f"conclusion classifier returned unknown IDs: {sorted(unknown)}")
    for node in nodes:
        if node["id"] in conclusion_ids:
            node["label"] = "conclusion"

    indices = [i for i, node in enumerate(nodes) if node["source"] == "response"]
    edge_fn = partial(rf.edge_detection_and_classification, nodes=nodes, llm_model_name=model)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        edge_lists = list(executor.map(edge_fn, indices))
    edges = [edge for group in edge_lists for edge in group]
    edges.sort(key=lambda edge: (rf.get_node_sort_key(edge["dest_node_id"]),
                                 rf.get_node_sort_key(edge["source_node_id"])))
    for i, edge in enumerate(edges):
        edge["id"] = f"e{i}"
    datum["edges"] = edges
    datum["metadata"]["annotator"] = model
    return datum


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces-root", type=Path, default=DEFAULT_TRACES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT / "reasoning_annotations")
    parser.add_argument("--debug-root", type=Path, default=DEFAULT_OUT / "debug" / "reasoning")
    parser.add_argument("--provider-profile", default="uvarc")
    parser.add_argument("--model", default="Kimi K2.5")
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--node-retry-attempts", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if (args.workers < 1 or args.max_tokens < 1 or args.limit < 0
            or args.node_retry_attempts < 1):
        parser.error("workers/max-tokens/node-retry-attempts must be positive and limit nonnegative")
    rf.configure_llm(profile=args.provider_profile, model=args.model,
                     max_tokens=args.max_tokens, debug_root=args.debug_root)
    attempted = failed = completed = skipped = 0
    sources = list(iter_trace_files(args.traces_root))
    if not sources:
        raise SystemExit(f"No trace files found under {args.traces_root}")
    for source in sources:
        if args.limit and attempted >= args.limit:
            break
        rel = source.relative_to(args.traces_root)
        doc_id = "__".join(rel.with_suffix("").parts)
        destination = args.output_dir / f"{doc_id}.json"
        if destination.exists() and not args.overwrite:
            skipped += 1
            continue
        attempted += 1
        try:
            doc = annotate_document(convert_trace(read_json(source), rel), args.model, args.workers,
                                    node_retry_attempts=args.node_retry_attempts)
            args.output_dir.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile("w", dir=args.output_dir, delete=False,
                                             suffix=".json", encoding="utf-8") as handle:
                import json
                json.dump(doc, handle, indent=2, ensure_ascii=False)
                handle.write("\n")
                temporary = Path(handle.name)
            os.replace(temporary, destination)
            completed += 1
            print(f"ok: {rel}", flush=True)
        except Exception as exc:
            failed += 1
            print(f"FAIL: {rel}: {type(exc).__name__}: {exc}", flush=True)
    print(f"attempted={attempted} completed={completed} skipped={skipped} failed={failed}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
