#!/usr/bin/env python3
"""Repair whitespace-only gaps between ReasoningFlow node source spans."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

from seabench_common import DEFAULT_OUT, read_json


def _assert_exact_slice(node: dict, raw: str) -> None:
    start, end = node.get("start"), node.get("end")
    if not isinstance(start, int) or not isinstance(end, int):
        raise ValueError(f"{node.get('id')}: non-integer source span")
    if not 0 <= start < end <= len(raw):
        raise ValueError(f"{node.get('id')}: invalid source span {start}:{end}")
    if node.get("text") != raw[start:end]:
        raise ValueError(f"{node.get('id')}: text differs from its source slice")


def repair_document(doc: dict) -> list[dict]:
    """Absorb whitespace-only uncovered spans into their adjacent nodes."""
    repairs: list[dict] = []
    raw_text = doc.get("raw_text") or {}
    for source in ("question", "response"):
        raw = str(raw_text.get(source, ""))
        nodes = sorted(
            (node for node in doc.get("nodes", []) if node.get("source") == source),
            key=lambda node: (node.get("start", -1), node.get("end", -1)),
        )
        if not nodes:
            if raw:
                raise ValueError(f"{source}: nonempty source has no nodes")
            continue
        for node in nodes:
            _assert_exact_slice(node, raw)

        first = nodes[0]
        if first["start"]:
            gap = raw[:first["start"]]
            if not gap.isspace():
                raise ValueError(f"{source}: non-whitespace leading gap")
            old_start = first["start"]
            first["start"] = 0
            first["text"] = raw[:first["end"]]
            repairs.append({"source": source, "position": "leading", "chars": old_start})

        for previous, current in zip(nodes, nodes[1:]):
            if previous["end"] > current["start"]:
                raise ValueError(
                    f"{source}: overlapping nodes {previous['id']} and {current['id']}"
                )
            if previous["end"] < current["start"]:
                gap = raw[previous["end"]:current["start"]]
                if not gap.isspace():
                    raise ValueError(
                        f"{source}: non-whitespace gap between "
                        f"{previous['id']} and {current['id']}"
                    )
                old_end = previous["end"]
                previous["end"] = current["start"]
                previous["text"] = raw[previous["start"]:previous["end"]]
                repairs.append({
                    "source": source, "position": "between",
                    "after": previous["id"], "before": current["id"],
                    "chars": current["start"] - old_end,
                })

        last = nodes[-1]
        if last["end"] < len(raw):
            gap = raw[last["end"]:]
            if not gap.isspace():
                raise ValueError(f"{source}: non-whitespace trailing gap")
            old_end = last["end"]
            last["end"] = len(raw)
            last["text"] = raw[last["start"]:]
            repairs.append({
                "source": source, "position": "trailing",
                "after": last["id"], "chars": len(raw) - old_end,
            })

        reconstructed = "".join(node["text"] for node in nodes)
        if reconstructed != raw:
            raise ValueError(f"{source}: repair did not reconstruct source exactly")
    return repairs


def atomic_write(path: Path, doc: dict) -> None:
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, delete=False, suffix=".json", encoding="utf-8"
    ) as handle:
        json.dump(doc, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotations-dir", type=Path,
        default=DEFAULT_OUT / "reasoning_annotations",
    )
    parser.add_argument(
        "--backup-dir", type=Path,
        help="Backup directory (default: OUT_ROOT/backup/text_partitions)",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.annotations_dir.is_dir():
        parser.error(f"annotations directory not found: {args.annotations_dir}")
    backup_dir = args.backup_dir or args.annotations_dir.parent / "backup" / "text_partitions"

    changed = unchanged = failed = 0
    for path in sorted(args.annotations_dir.glob("*.json")):
        doc = read_json(path)
        try:
            repairs = repair_document(doc)
        except Exception as exc:
            failed += 1
            print(f"FAIL: {path.name}: {type(exc).__name__}: {exc}")
            continue
        if not repairs:
            unchanged += 1
            continue
        repaired_chars = sum(item["chars"] for item in repairs)
        if args.dry_run:
            print(
                f"would repair: {path.name}: {len(repairs)} gap(s), "
                f"{repaired_chars} whitespace character(s)"
            )
        else:
            backup_dir.mkdir(parents=True, exist_ok=True)
            backup = backup_dir / path.name
            if not backup.exists():
                shutil.copy2(path, backup)
            atomic_write(path, doc)
            print(
                f"repaired: {path.name}: {len(repairs)} gap(s), "
                f"{repaired_chars} whitespace character(s)"
            )
        changed += 1
    print(
        f"changed={changed} unchanged={unchanged} failed={failed} "
        f"dry_run={args.dry_run} backups={backup_dir}"
    )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
