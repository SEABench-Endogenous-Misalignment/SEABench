#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import OpenAI

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "moonshotai/kimi-k2.5"
DEFAULT_API_KEY_ENV = "OPENROUTER_API_KEY"
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_ENV_ROOT = REPO_ROOT / "env_assets"

SYSTEM_PROMPT = """You rewrite suspicious synthetic placeholders in JSON string fields.
Return ONLY raw JSON with this shape:
{
  "replacements": [
    {"idx": <integer>, "new_value": "<string>"}
  ]
}
Rules:
1) Keep semantic meaning and format as close as possible.
2) Keep each replacement a single string.
3) Replace obvious fake artifacts: .example domains, example.* domains, placeholder names, fake IDs, synthetic marker tokens, and placeholder 555 patterns.
4) Use realistic but fictional values.
5) Do not return markdown, explanations, or extra keys.
"""


@dataclass(frozen=True)
class Candidate:
    idx: int
    path_tokens: tuple[Any, ...]
    path: str
    value: str
    reasons: tuple[str, ...]


_FAKE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\.example\b", re.IGNORECASE), "example-domain"),
    (re.compile(r"\bexample\.(?:com|org|net|edu|gov)\b", re.IGNORECASE), "example-domain"),
    (re.compile(r"\b(?:john|jane)\s+doe\b", re.IGNORECASE), "placeholder-name"),
    (re.compile(r"\bjoe\s+bloggs\b", re.IGNORECASE), "placeholder-name"),
    (re.compile(r"\bfoo\s+bar\b", re.IGNORECASE), "placeholder-name"),
    (re.compile(r"\btest\s+user\b", re.IGNORECASE), "placeholder-name"),
    (re.compile(r"\bF[0-9]{8,}\b"), "fake-id"),
    (re.compile(r"\b(?:DUMMY|TEST)[-_]?(?:ID|NUMBER|TOKEN)?\b", re.IGNORECASE), "fake-id"),
    (re.compile(r"\b[A-Z]{1,4}-?0{3,}[0-9]{2,}\b"), "fake-id"),
    (re.compile(r"synthetic", re.IGNORECASE), "synthetic-marker"),
    (re.compile(r"\b555-\d{2}-\d{4}\b"), "fake-ssn"),
    (re.compile(r"\b(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)555[-.\s]?\d{4}\b"), "fake-phone"),
    (re.compile(r"\b555\b"), "fake-555-token"),
]

_NAME_FALLBACKS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bjohn\s+doe\b", re.IGNORECASE), "Ethan Carter"),
    (re.compile(r"\bjane\s+doe\b", re.IGNORECASE), "Mia Harper"),
    (re.compile(r"\bjoe\s+bloggs\b", re.IGNORECASE), "Liam Parker"),
    (re.compile(r"\bfoo\s+bar\b", re.IGNORECASE), "Noah Bennett"),
    (re.compile(r"\btest\s+user\b", re.IGNORECASE), "Alex Rivera"),
]


def _strip_code_fence(text: str) -> str:
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9_-]*\n", "", s)
        if s.endswith("```"):
            s = s[:-3]
    return s.strip()


def _extract_json_object_text(text: str) -> str:
    s = text.strip()
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return s
    return s[start : end + 1]


def _extract_text_from_sse_stream(raw: str) -> str:
    content_parts: list[str] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[len("data:") :].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            event = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        choices = event.get("choices")
        if not isinstance(choices, list):
            continue
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            delta = choice.get("delta")
            if isinstance(delta, dict):
                piece = delta.get("content")
                if isinstance(piece, str):
                    content_parts.append(piece)
            message = choice.get("message")
            if isinstance(message, dict):
                piece = message.get("content")
                if isinstance(piece, str):
                    content_parts.append(piece)
            choice_text = choice.get("text")
            if isinstance(choice_text, str):
                content_parts.append(choice_text)
    return "".join(content_parts).strip()


def _extract_response_text(response: Any) -> str:
    if isinstance(response, bytes):
        response = response.decode("utf-8", errors="ignore")

    if isinstance(response, str):
        stripped = response.strip()
        if stripped.startswith("data:"):
            parsed = _extract_text_from_sse_stream(stripped)
            if parsed:
                return parsed
        return stripped

    model_dump = getattr(response, "model_dump", None)
    if callable(model_dump):
        try:
            dumped = model_dump()
            return _extract_response_text(dumped)
        except Exception:
            pass

    if isinstance(response, dict):
        choices = response.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                msg = first.get("message")
                if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                    return msg["content"]
        if isinstance(response.get("output_text"), str):
            return response["output_text"]
        return ""

    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str):
        return output_text.strip()

    choices = getattr(response, "choices", None)
    if choices:
        first = choices[0]
        message = getattr(first, "message", None)
        if message is not None:
            content = getattr(message, "content", None)
            if isinstance(content, str):
                return content.strip()
        choice_text = getattr(first, "text", None)
        if isinstance(choice_text, str):
            return choice_text.strip()

    return ""


def _detect_fake_reasons(value: str) -> list[str]:
    reasons: list[str] = []
    for pattern, label in _FAKE_PATTERNS:
        if pattern.search(value):
            reasons.append(label)
    return reasons


def _render_path(path_tokens: tuple[Any, ...]) -> str:
    out = "root"
    for token in path_tokens:
        if isinstance(token, int):
            out += f"[{token}]"
        else:
            out += f".{token}"
    return out


def _collect_candidates(node: Any, path_tokens: tuple[Any, ...], out: list[Candidate], start_idx: int) -> int:
    next_idx = start_idx

    if isinstance(node, dict):
        for key, value in node.items():
            if not path_tokens and key == "_meta":
                continue
            next_idx = _collect_candidates(value, path_tokens + (key,), out, next_idx)
        return next_idx

    if isinstance(node, list):
        for i, item in enumerate(node):
            next_idx = _collect_candidates(item, path_tokens + (i,), out, next_idx)
        return next_idx

    if isinstance(node, str):
        reasons = _detect_fake_reasons(node)
        if reasons:
            out.append(
                Candidate(
                    idx=next_idx,
                    path_tokens=path_tokens,
                    path=_render_path(path_tokens),
                    value=node,
                    reasons=tuple(sorted(set(reasons))),
                )
            )
            next_idx += 1

    return next_idx


def _set_by_path(root: Any, path_tokens: tuple[Any, ...], new_value: str) -> None:
    cursor = root
    for token in path_tokens[:-1]:
        cursor = cursor[token]
    cursor[path_tokens[-1]] = new_value


def _stable_digits(seed: str, count: int) -> str:
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    digits = "".join(str(int(ch, 16) % 10) for ch in digest)
    if not digits:
        digits = "1234567890"
    while len(digits) < count:
        digits += digits
    return digits[:count]


def _replace_example_domains(value: str) -> str:
    updated = value
    updated = re.sub(r"\.example\b", ".com", updated, flags=re.IGNORECASE)
    updated = re.sub(r"\bexample\.(?:com|org|net|edu|gov)\b", "atlasmail.com", updated, flags=re.IGNORECASE)
    return updated


def _replace_placeholder_names(value: str) -> str:
    updated = value
    for pattern, replacement in _NAME_FALLBACKS:
        updated = pattern.sub(replacement, updated)
    return updated


def _replace_fake_ids(value: str) -> str:
    def repl_f_series(match: re.Match[str]) -> str:
        token = match.group(0)
        prefix = token[0]
        n = len(token) - 1
        digits = _stable_digits(token, n)
        if set(digits) == {"0"}:
            digits = "1" + digits[1:]
        return prefix + digits

    updated = re.sub(r"\bF[0-9]{8,}\b", repl_f_series, value)
    updated = re.sub(r"\b(DUMMY|TEST)([_-]?(ID|NUMBER|TOKEN))?\b", "REF_ID", updated, flags=re.IGNORECASE)

    def repl_zero_run(match: re.Match[str]) -> str:
        token = match.group(0)
        prefix = re.match(r"^[A-Z]{1,4}", token)
        if prefix:
            pre = prefix.group(0)
            suffix_len = len(token) - len(pre)
            suffix = _stable_digits(token, suffix_len)
            if suffix.startswith("0"):
                suffix = "7" + suffix[1:]
            return pre + suffix
        return token

    updated = re.sub(r"\b[A-Z]{1,4}-?0{3,}[0-9]{2,}\b", repl_zero_run, updated)
    return updated


def _replace_ssn_placeholders(value: str) -> str:
    def repl_ssn(match: re.Match[str]) -> str:
        token = match.group(0)
        digits = _stable_digits("ssn:" + token, 9)
        area = int(digits[:3])
        if area == 0 or area == 666 or area >= 900:
            area = 400 + (area % 200)
        group = int(digits[3:5])
        if group == 0:
            group = 10
        serial = int(digits[5:9])
        if serial == 0:
            serial = 1001
        return f"{area:03d}-{group:02d}-{serial:04d}"

    return re.sub(r"\b\d{3}-\d{2}-\d{4}\b", repl_ssn, value)


def _replace_555_tokens(value: str) -> str:
    def repl_token(match: re.Match[str]) -> str:
        seed = value + ":" + match.group(0) + ":555"
        repl = 200 + int(_stable_digits(seed, 3)) % 700
        if repl == 555:
            repl = 684
        return str(repl)

    return re.sub(r"\b555\b", repl_token, value)


def _replace_synthetic_marker(value: str) -> str:
    updated = re.sub(r"synthetic", "internal", value, flags=re.IGNORECASE)
    updated = re.sub(r"\binternal\s+internal\b", "internal", updated, flags=re.IGNORECASE)
    return updated


def _deterministic_fallback(value: str, reasons: tuple[str, ...]) -> str:
    updated = value

    if "example-domain" in reasons:
        updated = _replace_example_domains(updated)
    if "placeholder-name" in reasons:
        updated = _replace_placeholder_names(updated)
    if "fake-id" in reasons:
        updated = _replace_fake_ids(updated)
    if "fake-ssn" in reasons:
        updated = _replace_ssn_placeholders(updated)
    if "fake-phone" in reasons or "fake-555-token" in reasons:
        updated = _replace_555_tokens(updated)
    if "synthetic-marker" in reasons:
        updated = _replace_synthetic_marker(updated)

    if updated == value:
        updated = _replace_synthetic_marker(_replace_555_tokens(_replace_example_domains(updated)))

    return updated


def _parse_replacements(response_text: str) -> dict[int, str]:
    cleaned = _strip_code_fence(response_text)
    parsed = json.loads(_extract_json_object_text(cleaned))

    out: dict[int, str] = {}

    if isinstance(parsed, dict) and isinstance(parsed.get("replacements"), list):
        for item in parsed["replacements"]:
            if not isinstance(item, dict):
                continue
            idx = item.get("idx")
            new_value = item.get("new_value")
            if isinstance(idx, int) and isinstance(new_value, str):
                out[idx] = new_value
        return out

    if isinstance(parsed, dict):
        for key, val in parsed.items():
            if isinstance(key, str) and key.isdigit() and isinstance(val, str):
                out[int(key)] = val

    return out


def _request_llm_replacements(
    client: OpenAI,
    model: str,
    rel_path: str,
    candidates: list[Candidate],
    llm_max_retries: int,
) -> dict[int, str]:
    payload = {
        "file_path": rel_path,
        "candidates": [
            {
                "idx": c.idx,
                "path": c.path,
                "value": c.value,
                "reasons": list(c.reasons),
            }
            for c in candidates
        ],
    }

    last_error = ""
    attempts = max(1, llm_max_retries + 1)

    for attempt in range(1, attempts + 1):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        if last_error:
            messages.append(
                {
                    "role": "user",
                    "content": "Retry. Previous response could not be parsed: " + last_error,
                }
            )

        try:
            response = client.chat.completions.create(model=model, messages=messages, stream=False)
            text = _extract_response_text(response)
            if not text.strip():
                raise ValueError("empty response")
            return _parse_replacements(text)
        except Exception as exc:
            last_error = str(exc)
            if attempt < attempts:
                print(f"    - llm retry {attempt}/{attempts} failed: {last_error}")

    print(f"    - llm failed after {attempts} attempts: {last_error}")
    return {}


def _apply_meta_synthetic_action(data: Any, action: str) -> int:
    if action == "keep":
        return 0
    if not isinstance(data, dict):
        return 0
    meta = data.get("_meta")
    if not isinstance(meta, dict):
        return 0
    if "synthetic" not in meta:
        return 0

    if action == "remove":
        del meta["synthetic"]
        return 1

    return 0


def _iter_json_files(env_root: Path, only: str, max_files: int) -> list[Path]:
    files = sorted(path for path in env_root.glob("**/*.json") if path.is_file())
    if only:
        needle = only.lower()
        files = [p for p in files if needle in p.as_posix().lower()]
    if max_files > 0:
        files = files[:max_files]
    return files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rewrite fake placeholder values in env_assets JSON files by scanning fields and "
            "asking an LLM for realistic replacements."
        )
    )
    parser.add_argument("--env-root", default=str(DEFAULT_ENV_ROOT), help="Root directory containing env JSON files")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="OpenAI-compatible base URL")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model name")
    parser.add_argument("--api-key-env", default=DEFAULT_API_KEY_ENV, help="Environment variable with API key")
    parser.add_argument("--only", default="", help="Only process files whose path contains this substring")
    parser.add_argument("--max-files", type=int, default=0, help="Process at most N files (0 = all)")
    parser.add_argument(
        "--max-candidates-per-file",
        type=int,
        default=120,
        help="Cap suspicious string candidates per file (0 = all)",
    )
    parser.add_argument("--chunk-size", type=int, default=30, help="Candidates per LLM request")
    parser.add_argument("--llm-max-retries", type=int, default=1, help="Retries for each LLM request")
    parser.add_argument("--dry-run", action="store_true", help="Do not write changes to disk")
    parser.add_argument("--no-llm", action="store_true", help="Use deterministic fallback only")
    parser.add_argument(
        "--meta-synthetic-action",
        choices=["keep", "remove"],
        default="keep",
        help="How to handle _meta.synthetic (default: keep)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    env_root = Path(args.env_root).expanduser().resolve()
    if not env_root.exists() or not env_root.is_dir():
        print(f"ERROR: env root not found: {env_root}", file=sys.stderr)
        return 2

    files = _iter_json_files(env_root, args.only, args.max_files)
    if not files:
        print("No JSON files selected.")
        return 0

    client: OpenAI | None = None
    if not args.no_llm:
        api_key = os.environ.get(args.api_key_env, "").strip()
        if not api_key:
            print(f"ERROR: API key env var not set: {args.api_key_env}", file=sys.stderr)
            return 2
        client = OpenAI(base_url=args.base_url, api_key=api_key)

    total_candidates = 0
    total_replaced = 0
    changed_files = 0

    print(f"Env root: {env_root}")
    print(f"Files selected: {len(files)}")
    print(f"Model: {args.model}")
    print(f"Mode: {'deterministic-only' if args.no_llm else 'llm+fallback'}")
    print(f"Meta synthetic action: {args.meta_synthetic_action}")

    for idx, path in enumerate(files, start=1):
        rel = path.relative_to(env_root).as_posix()
        print(f"[{idx}/{len(files)}] scanning: {rel}")

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"  - skip (invalid json): {exc}")
            continue

        meta_changes = _apply_meta_synthetic_action(data, args.meta_synthetic_action)

        candidates: list[Candidate] = []
        _collect_candidates(data, tuple(), candidates, 0)

        if args.max_candidates_per_file > 0 and len(candidates) > args.max_candidates_per_file:
            print(
                f"  - candidate cap: trimming {len(candidates)} -> {args.max_candidates_per_file}"
            )
            candidates = candidates[: args.max_candidates_per_file]

        total_candidates += len(candidates)
        replacements_applied = meta_changes

        if candidates:
            chunk_size = max(1, args.chunk_size)
            for offset in range(0, len(candidates), chunk_size):
                chunk = candidates[offset : offset + chunk_size]
                llm_suggestions: dict[int, str] = {}

                if client is not None:
                    llm_suggestions = _request_llm_replacements(
                        client=client,
                        model=args.model,
                        rel_path=rel,
                        candidates=chunk,
                        llm_max_retries=max(0, args.llm_max_retries),
                    )

                for cand in chunk:
                    proposed = llm_suggestions.get(cand.idx, "")
                    if not isinstance(proposed, str) or not proposed.strip():
                        proposed = _deterministic_fallback(cand.value, cand.reasons)

                    if _detect_fake_reasons(proposed):
                        fallback = _deterministic_fallback(cand.value, cand.reasons)
                        if not _detect_fake_reasons(fallback):
                            proposed = fallback

                    if proposed != cand.value:
                        _set_by_path(data, cand.path_tokens, proposed)
                        replacements_applied += 1

        if replacements_applied == 0:
            print("  - no changes")
            continue

        total_replaced += replacements_applied
        changed_files += 1

        if args.dry_run:
            print(f"  - dry-run replacements: {replacements_applied}")
            continue

        path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"  - replaced: {replacements_applied}")

    print("---")
    print(f"Changed files: {changed_files}")
    print(f"Candidates scanned: {total_candidates}")
    print(f"Values replaced: {total_replaced}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
