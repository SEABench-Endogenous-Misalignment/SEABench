#!/usr/bin/env python3
"""Step 1 of the safety-relevant CoT reasoning-pattern pipeline.

Extract the *selected-candidate* downstream reasoning traces (both the `none` control
arm and the `self_evolving` arm) for the three short-term-memory models, into one
self-contained JSON per (model, harm-type, arm, task).

Only downstream tasks are pulled (``optimization_summary.json`` ``role == "downstream"``;
in practice tasks 006-010). For each we read the reasoning trace of the last attempt of
the selected candidate's replay, plus the hidden safety outcome.

Each trace JSON also carries `units`: the deterministic rule pre-pass segmentation
(turn / paragraph / sentence split) used by the one-shot adapted annotator. The
original staged ReasoningFlow adapter consumes the same trace JSON but performs its
own original LLM segmentation and alignment. Downstream annotation pipelines consume
the JSON written here, so they never touch the raw run tree again.

Usage:
  python3 prompt_optimization_analysis/cot_extract_traces.py \
    --models kimi_k2_5_short_term_memory --harmtypes hallucination
  python3 prompt_optimization_analysis/cot_extract_traces.py   # all 3 models, all harm types

Output:
  prompt_optimization/out/cot_safety_patterns/traces/<model>/<category>_<harmtype>/<arm>/task_0XX.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

_STATS: Counter = Counter()

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = REPO_ROOT / "prompt_optimization" / "out"
DEFAULT_DEST = OUT_ROOT / "cot_safety_patterns" / "traces"

DEFAULT_MODELS = (
    "gpt_5.6_luna_short_term_memory",
    "grok_4.3_short_term_memory",
    "kimi_k2_5_short_term_memory",
)

# Harm-type suffixes, longest first so "boundary_collapse" wins over a bare "collapse".
KNOWN_HARMTYPES = (
    "contextual_boundary_collapse",
    "boundary_collapse",
    "guardrail_erosion",
    "hallucination",
    "privacy",
)

ARMS = ("none", "self_evolving")
ARM_REPLAY_KEY = {"none": "best_none_replay_dir", "self_evolving": "best_self_replay_dir"}
SURFACE_NAMES = {
    "short_term_memory": "short_term_memory",
    "controller_update": "controller_update",
    "tool-use": "tool_use",
}


# ---------------------------------------------------------------------------
# run-tree helpers (kept local so this script has no heavy imports)
# ---------------------------------------------------------------------------

def _numbered_task_dirs(cascade_dir: Path) -> list[Path]:
    dirs = [d for d in cascade_dir.iterdir() if d.is_dir() and re.match(r"^\d+_task_\d+$", d.name)]
    return sorted(dirs, key=lambda d: int(d.name.split("_")[0]))


def _downstream_run_dirs(search_root: Path) -> list[Path]:
    """Find run dirs anywhere beneath a source root."""
    if not search_root.is_dir():
        return []
    matches: list[Path] = []
    for parent, dirnames, _filenames in os.walk(search_root):
        parent_path = Path(parent)
        if (parent_path / "cascade_optimization").is_dir():
            matches.append(parent_path)
            # A matched directory is itself a complete run tree. Pruning it
            # avoids recursively scanning its potentially large evaluation
            # artifacts and avoids nesting false positives inside it.
            dirnames[:] = []
    return sorted(matches)


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _selected_candidate_from_log(task_dir: Path) -> str:
    log_path = task_dir / "optimization_log.jsonl"
    if not log_path.exists():
        return "current"
    selected = "current"
    for line in log_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if entry.get("selected") and entry.get("candidate"):
            selected = str(entry["candidate"])
    return selected


def _glob_replay_dir(task_dir: Path, candidate: str, arm: str) -> Path | None:
    pattern = str(task_dir / "evaluations" / candidate / arm / "*")
    matches = sorted(p for p in glob.glob(pattern) if (Path(p) / "task_result.json").exists())
    return Path(matches[-1]) if matches else None


def _cascade_root(path: Path) -> Path | None:
    try:
        idx = path.parts.index("cascade_optimization")
    except ValueError:
        return None
    return Path(*path.parts[: idx + 1])


def _relocate_cascade_path(path: Path, cascade_root: Path | None) -> Path:
    """Remap a stale absolute path (e.g. recorded under another user's
    scratch dir before this output tree was copied/moved) onto the current
    cascade_root, when the original path is missing or unreadable."""
    try:
        if path.exists():
            return path
    except PermissionError:
        pass
    if cascade_root is None:
        return path
    try:
        cascade_idx = path.parts.index("cascade_optimization")
    except ValueError:
        return path
    relocated = cascade_root.joinpath(*path.parts[cascade_idx + 1 :])
    try:
        relocated_exists = relocated.exists()
    except PermissionError:
        relocated_exists = False
    if relocated_exists:
        _STATS["relocated"] += 1
        return relocated
    return path


def _resolve_arm_replay(task_dir: Path, summary: dict[str, Any] | None, candidate: str, arm: str) -> Path | None:
    if summary is not None:
        rd = summary.get(ARM_REPLAY_KEY[arm])
        if rd:
            rd_path = _relocate_cascade_path(Path(rd), _cascade_root(task_dir))
            try:
                if rd_path.exists() and (rd_path / "task_result.json").exists():
                    return rd_path
            except PermissionError:
                pass
    return _glob_replay_dir(task_dir, candidate, arm)


def parse_run_dir_name(name: str) -> tuple[str, str] | None:
    """Best-effort split of a run dir name into (category, harmtype)."""
    stem = name
    if stem.endswith("_downstream_rerun"):
        stem = stem[: -len("_downstream_rerun")]
    stem = stem.replace("-", "_")

    # Explicit double-underscore delimited fields (e.g. name__category__harmtype__model__date)
    # unambiguously separate tokens, so prefer an exact field match when present.
    if "__" in stem:
        fields = [f for f in stem.split("__") if f]
        for ht in KNOWN_HARMTYPES:
            for i, field in enumerate(fields):
                if field.lower() == ht and i > 0:
                    return fields[i - 1], ht

    # Fall back to scanning underscore-separated tokens for a contiguous run
    # matching a known harmtype, wherever it occurs in the name.
    tokens = [t for t in stem.split("_") if t]
    for ht in KNOWN_HARMTYPES:
        ht_tokens = ht.split("_")
        n = len(ht_tokens)
        for i in range(1, len(tokens) - n + 1):
            if tokens[i:i + n] == ht_tokens:
                return "_".join(tokens[:i]), ht

    return None


# ---------------------------------------------------------------------------
# deterministic rule pre-pass segmentation (coarse units; an LLM stage later
# refines multi-role units into atomic nodes -- see annotate.py)
# ---------------------------------------------------------------------------

_PROTECT = [
    r"e\.g\.", r"i\.e\.", r"etc\.", r"vs\.", r"cf\.", r"No\.", r"al\.",
    r"Mr\.", r"Mrs\.", r"Ms\.", r"Dr\.", r"Prof\.", r"Sr\.", r"Jr\.", r"St\.",
    r"[A-Z]\.",                       # single-letter initials
    r"\d+\.\d+",                      # decimals
    r"\b\w+\.(?:json|md|txt|yaml|yml|csv|py|jsonl)\b",  # filenames
]
_PROTECT_RE = re.compile("(" + "|".join(_PROTECT) + ")")
_SENT_SPLIT_RE = re.compile(r"(?<=[.?!])\s+(?=[\"'\[(]?[A-Z0-9])")
_BULLET_RE = re.compile(r"^\s*(?:[-*•]\s+|\d+[.)]\s+)")
_LONG = 600


def _mask(text: str) -> tuple[str, list[str]]:
    stash: list[str] = []

    def repl(m: re.Match) -> str:
        stash.append(m.group(0))
        return f"\x00{len(stash) - 1}\x00"

    return _PROTECT_RE.sub(repl, text), stash


def _unmask(text: str, stash: list[str]) -> str:
    return re.sub(r"\x00(\d+)\x00", lambda m: stash[int(m.group(1))], text)


def _split_paragraphs(block: str) -> list[str]:
    raw = re.split(r"\n{2,}", block)
    out: list[str] = []
    for chunk in raw:
        chunk = chunk.strip("\n")
        if not chunk.strip():
            continue
        if out and not (out[-1].rstrip().endswith((".", "?", "!", ":")) and re.match(r"[A-Z0-9\-*#]", chunk.lstrip())):
            out[-1] = out[-1] + "\n" + chunk
        else:
            out.append(chunk)
    return out


def _split_sentences(paragraph: str) -> list[str]:
    units: list[str] = []
    for line in paragraph.split("\n"):
        line = line.strip()
        if not line:
            continue
        if _BULLET_RE.match(line):
            units.append(line)
            continue
        masked, stash = _mask(line)
        for piece in _SENT_SPLIT_RE.split(masked):
            piece = _unmask(piece, stash).strip()
            if piece:
                units.append(piece)
    return units


def _split_long(unit: str) -> list[str]:
    if len(unit) <= _LONG or re.search(r"[.?!]\s*$", unit):
        return [unit]
    parts = [p.strip() for p in re.split(r"\s*(?:;|\s–\s|\s-\s)\s*", unit) if p.strip()]
    return parts or [unit]


def coarse_segment(turns: list[dict]) -> list[dict]:
    """turns: [{turn, reasoning}] -> [{index, turn, text}] coarse units."""
    units: list[dict] = []
    for t in turns:
        reasoning = str(t.get("reasoning") or "").strip()
        if not reasoning:
            continue
        turn = t.get("turn")
        for para in _split_paragraphs(reasoning):
            for sent in _split_sentences(para):
                for piece in _split_long(sent):
                    units.append({"index": len(units), "turn": turn, "text": piece})
    return units


# ---------------------------------------------------------------------------
# trace extraction
# ---------------------------------------------------------------------------

def _turns_from_attempt(attempt: dict[str, Any]) -> list[dict[str, Any]]:
    traces = ((attempt.get("task_run") or {}).get("reasoning_traces")) or []
    out: list[dict[str, Any]] = []
    for entry in traces:
        reasoning = str(entry.get("reasoning") or "").strip()
        out.append({"turn": entry.get("turn"), "reasoning": reasoning})
    return out


def _cot_text(turns: list[dict[str, Any]]) -> str:
    return "\n\n".join(
        f"[Turn {t['turn']}]\n{t['reasoning']}" for t in turns if t["reasoning"]
    )


def extract_one(
    task_dir: Path, arm: str, model: str, harmtype: str, category: str, surface: str,
) -> dict[str, Any] | None:
    summary = _load_json(task_dir / "optimization_summary.json")
    role = str((summary or {}).get("role") or "").lower()
    if role and role != "downstream":
        return None
    candidate = str((summary or {}).get("selected_candidate") or "") or _selected_candidate_from_log(task_dir)

    replay_dir = _resolve_arm_replay(task_dir, summary, candidate, arm)
    if replay_dir is None:
        _STATS["skip_no_replay_dir"] += 1
        return None

    result = _load_json(replay_dir / "task_result.json")
    if result is None:
        _STATS["skip_no_task_result"] += 1
        return None
    attempts = result.get("attempts")
    if not isinstance(attempts, list) or not attempts:
        _STATS["skip_no_attempts"] += 1
        return None
    last = attempts[-1]
    turns = _turns_from_attempt(last)

    safety = _load_json(replay_dir / "safety_result.json") or {}
    task_id = str(result.get("task_id") or task_dir.name.split("_", 1)[-1])

    return {
        "meta": {
            "model": model,
            "surface": surface,
            "harmtype": harmtype,
            "category": category,
            "task_id": task_id,
            "task_dir": task_dir.name,
            "arm": arm,
            "selected_candidate": candidate,
            "replay_dir": str(replay_dir),
            "attempt": last.get("attempt"),
            "n_attempts": len(attempts),
            "task_passed": last.get("passed", result.get("passed")),
        },
        "task_prompt": str(result.get("task_prompt") or last.get("task_prompt") or ""),
        "answer": str(result.get("answer") or last.get("answer") or ""),
        "safety_enabled": bool(safety.get("enabled", False)),
        "safety_passed": safety.get("passed"),
        "safety_mean_score": safety.get("mean_score"),
        "safety_scores": safety.get("scores"),
        "safety_feedback": safety.get("feedback", ""),
        "n_turns": len(turns),
        "n_turns_nonempty": sum(1 for t in turns if t["reasoning"]),
        "turns": turns,
        "cot_text": _cot_text(turns),
        "units": coarse_segment(turns),
    }


def run(
    models: list[str],
    harmtypes: list[str] | None,
    dest: Path,
    surface: str = "short_term_memory",
    source_roots: list[Path] | None = None,
) -> None:
    if source_roots:
        targets = []
        for source_root in source_roots:
            source_root = source_root.resolve()
            model_root = source_root.parent if source_root.name == "downstream_reruns" else source_root
            targets.append((model_root.name, source_root))
    else:
        targets = [
            (model, OUT_ROOT / model / "downstream_reruns")
            for model in models
        ]

    # Phase 1: locate every run dir up front, so extraction runs uninterrupted.
    resolved: list[tuple[str, Path, str, str]] = []
    unparsed: list[Path] = []
    missing_cascade: list[Path] = []
    for model, source_root in targets:
        if not source_root.is_dir():
            print(f"[warn] source root not found: {source_root}")
            continue
        for run_dir in _downstream_run_dirs(source_root):
            parsed = parse_run_dir_name(run_dir.name)
            if parsed is None:
                unparsed.append(run_dir)
                continue
            category, harmtype = parsed
            if harmtypes and harmtype not in harmtypes:
                continue
            if not (run_dir / "cascade_optimization").is_dir():
                missing_cascade.append(run_dir)
                continue
            resolved.append((model, run_dir, category, harmtype))
    print(
        f"Located {len(resolved)} run dir(s) across {len(targets)} source root(s) "
        f"({len(unparsed)} unparsed, {len(missing_cascade)} missing cascade_optimization/)."
    )

    # Phase 2: extract.
    n_written = 0
    for model, run_dir, category, harmtype in resolved:
        cascade = run_dir / "cascade_optimization"
        for task_dir in _numbered_task_dirs(cascade):
            for arm in ARMS:
                rec = extract_one(task_dir, arm, model, harmtype, category, surface)
                if rec is None:
                    continue
                out_fp = dest / model / f"{category}_{harmtype}" / arm / f"{rec['meta']['task_id']}.json"
                out_fp.parent.mkdir(parents=True, exist_ok=True)
                out_fp.write_text(json.dumps(rec, indent=2), encoding="utf-8")
                n_written += 1
                if not rec["n_turns_nonempty"]:
                    _STATS["no_reasoning_text"] += 1
    print("Extraction complete.")

    # Phase 3: summary.
    print(f"\nWrote {n_written} trace files under {dest}")
    if unparsed:
        print(f"  skipped (unparsed run dir name): {len(unparsed)} -> {[p.name for p in unparsed]}")
    if missing_cascade:
        print(f"  skipped (no cascade_optimization/): {len(missing_cascade)} -> {[p.name for p in missing_cascade]}")
    for key, count in sorted(_STATS.items()):
        print(f"  {key}: {count}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="*", default=list(DEFAULT_MODELS),
                    help="Model output-dir names under prompt_optimization/out/ (default: the 3 STM models)")
    ap.add_argument("--source-root", action="append", type=Path,
                    help="Root containing *_downstream_rerun directories; may be repeated. "
                         "When supplied, this takes precedence over --models.")
    ap.add_argument("--harmtypes", nargs="*", default=None,
                    help="Restrict to these harm types (default: all found)")
    ap.add_argument("--dest", type=Path, default=DEFAULT_DEST, help=f"Output root (default: {DEFAULT_DEST})")
    ap.add_argument(
        "--surface", choices=tuple(SURFACE_NAMES), default="short_term_memory",
        help="Experimental self-evolution surface recorded in each trace",
    )
    args = ap.parse_args()
    run(
        args.models, args.harmtypes, args.dest.resolve(),
        SURFACE_NAMES[args.surface], args.source_root,
    )


if __name__ == "__main__":
    main()
