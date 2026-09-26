"""UVARC/Kimi backend implementing ReasoningFlow's original LLM interface."""
from __future__ import annotations

import json
import re
import sys
import threading
import time
from pathlib import Path

SEABENCH_ROOT = Path(__file__).resolve().parents[4]

_profile = "uvarc"
_model = "Kimi K2.5"
_max_tokens = 32768
_debug_root: Path | None = None
_client = None
_extract = None
_opts = None
_lock = threading.Lock()
_counter = 0
in_token = 0
out_token = 0


def configure(*, profile: str = "uvarc", model: str = "Kimi K2.5",
              max_tokens: int = 32768, debug_root: Path | None = None) -> None:
    global _profile, _model, _max_tokens, _debug_root, _client, _extract, _opts, _counter
    _profile, _model, _max_tokens, _debug_root = profile, model, max_tokens, debug_root
    if _debug_root is not None:
        existing = []
        for path in _debug_root.glob("call_*.json"):
            match = re.fullmatch(r"call_(\d+)\.json", path.name)
            if match:
                existing.append(int(match.group(1)))
        _counter = max(existing, default=-1) + 1
    sys.path.insert(0, str(SEABENCH_ROOT / "prompt_optimization"))
    from refine_task_sequence_textgrad import (
        _extract_text_from_openai_string,
        _load_provider,
        _make_openai_client,
        _model_request_options,
    )
    provider = _load_provider(profile)
    _client = _make_openai_client(provider)
    _extract = _extract_text_from_openai_string
    _opts = _model_request_options(provider, model)


def _parse(content: str):
    value = (content or "").strip()
    if value.startswith("```"):
        value = re.sub(r"^```(?:json)?\s*", "", value)
        value = re.sub(r"\s*```$", "", value)
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        match = re.search(r"(\{.*\}|\[.*\])", value, re.DOTALL)
        return json.loads(match.group(1)) if match else None


def call_llm(prompt: str, schema=None, llm_model_name=None, thinking_level="minimal"):
    del thinking_level  # Kimi has no equivalent switch; stages remain separate.
    global _client, _counter
    if _client is None:
        configure(model=llm_model_name or _model)
    model = llm_model_name or _model
    raw = []
    parsed = None
    for attempt in range(4):
        messages = [{"role": "user", "content": prompt}]
        if attempt:
            messages.append({"role": "user", "content": "Return exactly one valid JSON value. No prose."})
        try:
            response = _client.chat.completions.create(
                model=model, messages=messages, temperature=0,
                max_tokens=_max_tokens, stream=False,
                response_format={"type": "json_object"}, **(_opts or {}),
            )
            content = _extract(response) if isinstance(response, str) else (response.choices[0].message.content or "")
        except Exception as exc:
            content = f"__ERROR__ {type(exc).__name__}: {exc}"
            time.sleep(2 + 2 * attempt)
        raw.append(content)
        try:
            parsed = _parse(content)
        except (json.JSONDecodeError, TypeError):
            parsed = None
        if parsed is not None:
            break
    with _lock:
        index = _counter
        _counter += 1
    if _debug_root is not None:
        path = _debug_root / f"call_{index:07d}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"model": model, "prompt": prompt, "raw": raw}, indent=2), encoding="utf-8")
    if parsed is None:
        raise RuntimeError("UVARC call failed to return JSON after 4 attempts")
    if schema is None:
        return parsed
    return schema.model_validate(parsed).model_dump(mode="json")


def get_metadata():
    return {"in_token": in_token, "out_token": out_token, "price": 0.0, "calls": _counter}
