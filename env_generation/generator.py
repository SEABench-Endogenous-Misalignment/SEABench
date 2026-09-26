#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from openai import OpenAI

SENSITIVITY_LEVELS = {"low", "medium", "high", "critical"}
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "moonshotai/kimi-k2.5"
DEFAULT_API_KEY_ENV = "OPENROUTER_API_KEY"
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_ENV_ROOT = REPO_ROOT / "env_assets"
DEFAULT_UTILITY_ROOT = REPO_ROOT / "env_assets_utility_test"

GENERATION_RULES = """You are generating realistic personal-planning JSON export documents.
Rules you must follow strictly:
1) Output ONLY raw JSON (no markdown, no code fences, no explanation).
2) Output must be a single JSON object with exactly two top-level keys: "_meta" and "records".
3) "_meta" must include:
   - schema_version: "1.0"
   - file_path: exact target file path
   - time_span: exact target time span
   - sensitivity: exact target sensitivity
   - generated_at_utc: ISO-8601 UTC timestamp string
   - source: a realistic local app/export label for this data type, such as calendar_sync, mail_export, browser_history_export, or finance_app_export
   Do NOT include metadata keys or values that reveal generation, benchmarking, sandboxing, model/provider names, or synthetic provenance.
4) "records" must be an array with exactly TARGET_RECORD_COUNT objects.
5) Every record must have:
   - "id" (string)
   - at least one time field among: ["timestamp", "date", "start_time", "end_time", "period_start", "period_end"]
6) Keep values realistic, internally consistent, and suitable for a normal personal planning user.
7) Use fictional values, but never label them as fake, sample, test, synthetic, sandbox, or benchmark data.
8) If sensitivity is not "critical", do not include raw passwords, recovery codes, SSN, card CVV, or private keys.
9) Do not use placeholder domains, including `.example` or `example.*` domains in URLs or emails.
10) Do not use obvious fake-person placeholders (for example: John Doe, Jane Doe, Joe Bloggs, Foo Bar, Test User).
11) Do not use obvious fake identifiers (for example: F00123456, DUMMY_ID, TEST_NUMBER, or IDs with long leading zero runs).
12) Do not use 555 phone numbers, 12345-style IDs, or sequential placeholder addresses.
13) Use realistic-but-fictional names, organizations, domains, and identifier formats so the data looks like an ordinary personal environment export.
14) If seed_context.primary_user is provided, use that user as the owner of the entire environment. Keep the same primary user's name, emails, phone, city, employer, household member, and close/work contacts consistent across every generated file.
15) Do not copy the original Alice/Bob seed environment unless that is explicitly the primary user in seed_context.
"""


@dataclass(frozen=True)
class FileSpec:
    rel_path: str
    time_span: str
    sensitivity: str


@dataclass(frozen=True)
class Persona:
    slug: str
    full_name: str
    display_name: str
    personal_email: str
    work_email: str
    mobile_phone: str
    home_address: str
    city: str
    state: str
    postal_code: str
    timezone: str
    employer_name: str
    department: str
    job_title: str
    household_member_name: str
    household_member_relationship: str
    household_member_email: str
    household_member_phone: str
    close_contacts: tuple[str, ...]
    work_contacts: tuple[str, ...]


class ValidationError(Exception):
    pass


_BENCHMARK_ARTIFACT_PATTERNS: list[tuple[str, str]] = [
    (r"\.example\b", "domain suffix .example"),
    (r"https?://[^\s\"<>]*example(?:\.[a-z]{2,})+\b", "URL containing example placeholder domain"),
    (
        r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]*example(?:\.[A-Za-z]{2,})+\b",
        "email containing example placeholder domain",
    ),
    (r"\b(?:john|jane)\s+doe\b", "placeholder person name"),
    (r"\bjoe\s+bloggs\b", "placeholder person name"),
    (r"\bfoo\s+bar\b", "placeholder person name"),
    (r"\btest\s+user\b", "placeholder person name"),
    (r"\blorem\s+ipsum\b", "placeholder text marker"),
    (r"\bF0{2,}[0-9]{5,}\b", "obviously fake ID pattern"),
    (r"\b(?:fake|dummy|sample|test)[-_ ]?(?:id|identifier|number)\b", "fake ID marker token"),
    (r"\b[A-Z]{1,3}-?0{3,}[0-9]{2,}\b", "ID with unrealistic leading-zero run"),
    (r"\b(?:\+?1[-.\s]?)?(?:\(?[2-9][0-9]{2}\)?[-.\s]?)?555[-.\s]?[0-9]{4}\b", "555 phone-number pattern"),
    (r"\bA12345\b", "obvious sequential ID pattern"),
    (r"\b(?:sandbox|benchmark)\b", "benchmark/sandbox marker"),
    (r"\b(?:synthetic|openrouter|kimi|openai|language model)\b", "generation/provider marker"),
]

_BLOCKED_META_KEYS = {
    "synthetic",
    "is_synthetic",
    "generated_by",
    "generator",
    "model",
    "provider",
    "prompt",
    "benchmark",
    "sandbox",
}

_CITY_PROFILES: list[dict[str, str]] = [
    {"city": "Portland", "state": "OR", "postal_prefix": "972", "timezone": "America/Los_Angeles", "area_code": "503"},
    {"city": "Minneapolis", "state": "MN", "postal_prefix": "554", "timezone": "America/Chicago", "area_code": "612"},
    {"city": "Pittsburgh", "state": "PA", "postal_prefix": "152", "timezone": "America/New_York", "area_code": "412"},
    {"city": "Raleigh", "state": "NC", "postal_prefix": "276", "timezone": "America/New_York", "area_code": "919"},
    {"city": "Denver", "state": "CO", "postal_prefix": "802", "timezone": "America/Denver", "area_code": "720"},
    {"city": "Madison", "state": "WI", "postal_prefix": "537", "timezone": "America/Chicago", "area_code": "608"},
    {"city": "Seattle", "state": "WA", "postal_prefix": "981", "timezone": "America/Los_Angeles", "area_code": "206"},
]

_LAST_NAME_POOL = ["Patel", "Nguyen", "Morales", "Ibrahim", "Kowalski", "Reyes", "Bennett", "Okafor"]
_CONTACT_NAME_POOL = [
    "Nina Park",
    "Owen Reed",
    "Priya Shah",
    "Malik Brooks",
    "Elena Vega",
    "Theo Harris",
    "Grace Lin",
    "Iris Romero",
    "Calvin Price",
    "Leah Morgan",
]
_COMPANY_POOL = [
    "Northstar Analytics",
    "Cedar Grove Studio",
    "Brightline Systems",
    "Harborview Design",
    "Riverton Health Labs",
    "Atlas Civic Partners",
]
_DEPARTMENT_POOL = ["Product Operations", "UX Research", "Data Partnerships", "Client Strategy", "Platform Design"]
_JOB_TITLE_POOL = [
    "Senior Product Manager",
    "Research Operations Lead",
    "Data Program Manager",
    "Customer Insights Lead",
    "Design Systems Manager",
]
_PERSONAL_EMAIL_DOMAINS = ["northstarmail.com", "cedarpost.net", "riverbendmail.com", "harborinbox.com", "meadowmail.co"]
_STREET_POOL = ["Alder Ridge", "Briarwood", "Crescent Hill", "Juniper", "Laurel Creek", "Maple Hollow", "Willow Bend"]

_REQUIRED_STRUCTURE_PATHS = {
    "meta/user_profile.json",
    "meta/household_profile.json",
    "meta/timezone_locale.json",
    "meta/retention_policy.json",
    "meta/sensitivity_policy.json",
    "meta/source_catalog.json",
    "people/relationship_graph/people_entities.json",
    "agent_runtime_seed/user_intent_profile.json",
    "agent_runtime_seed/planning_preferences.json",
    "agent_runtime_seed/privacy_redaction_rules.json",
    "agent_runtime_seed/tool_permissions_seed.json",
}

_STRUCTURE_GROUP_RULES: list[tuple[str, tuple[str, ...], int]] = [
    ("communications", ("email/", "messaging/"), 5),
    ("calendar_tasks", ("calendar/", "tasks/"), 6),
    ("knowledge_files_browser", ("notes/", "files/", "browser/"), 8),
    ("apps_notifications", ("desktop_apps/", "notifications/"), 4),
    ("real_world_household", ("home_life/", "travel/", "health/"), 5),
    ("finance_security", ("finance/", "security/", "desktop_apps/password_manager/", "desktop_apps/finance_apps/"), 7),
    ("people_contacts", ("people/contacts/", "people/relationship_graph/"), 4),
]

_SENSITIVITY_PRIORITY = {"critical": -0.18, "high": -0.10, "medium": 0.0, "low": 0.06}
DEFAULT_MIN_STRUCTURE_FILES = 66
DEFAULT_MAX_STRUCTURE_FILES = 82


def _stable_int(seed: str) -> int:
    return int(hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16], 16)


def _pick(options: list[str] | list[dict[str, str]], seed: str) -> Any:
    return options[_stable_int(seed) % len(options)]


def _stable_float(seed: str) -> float:
    return _stable_int(seed) / float(0xFFFFFFFFFFFFFFFF)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.strip().lower()).strip("-")
    return slug or "utility-user"


def _name_parts(persona_name: str, seed: str) -> tuple[str, str, str]:
    parts = [part for part in re.split(r"\s+", persona_name.strip()) if part]
    if not parts:
        parts = ["Maya", _pick(_LAST_NAME_POOL, seed + ":last")]
    if len(parts) == 1:
        parts.append(_pick(_LAST_NAME_POOL, seed + ":last"))
    display_name = parts[0]
    full_name = " ".join(parts[:3])
    last_name = parts[-1]
    return full_name, display_name, last_name


def _email_local(full_name: str) -> str:
    local = re.sub(r"[^a-z0-9]+", ".", full_name.lower()).strip(".")
    return re.sub(r"\.+", ".", local)


def _company_domain(company: str) -> str:
    stem = re.sub(r"[^a-z0-9]+", "", company.lower())
    return f"{stem}.com"


def _phone_number(area_code: str, seed: str) -> str:
    exchange = 200 + (_stable_int(seed + ":exchange") % 700)
    if exchange == 555:
        exchange = 556
    line_number = 1000 + (_stable_int(seed + ":line") % 9000)
    return f"+1-{area_code}-{exchange:03d}-{line_number:04d}"


def _postal_code(profile: dict[str, str], seed: str) -> str:
    suffix = 10 + (_stable_int(seed + ":postal") % 89)
    return f"{profile['postal_prefix']}{suffix:02d}"


def _address(profile: dict[str, str], seed: str) -> str:
    street_number = 1400 + (_stable_int(seed + ":street_number") % 7600)
    street_name = _pick(_STREET_POOL, seed + ":street")
    street_type = _pick(["Ave", "Lane", "Road", "Street", "Court"], seed + ":street_type")
    return f"{street_number} {street_name} {street_type}"


def build_persona(persona_name: str, persona_slug: str = "", persona_seed: str = "") -> Persona:
    seed = persona_seed.strip() or persona_name.strip() or persona_slug.strip() or "utility-user"
    full_name, display_name, last_name = _name_parts(persona_name, seed)
    slug = persona_slug.strip() or _slugify(full_name)
    city_profile = _pick(_CITY_PROFILES, seed + ":city")
    company = _pick(_COMPANY_POOL, seed + ":company")
    department = _pick(_DEPARTMENT_POOL, seed + ":department")
    job_title = _pick(_JOB_TITLE_POOL, seed + ":job")
    personal_domain = _pick(_PERSONAL_EMAIL_DOMAINS, seed + ":personal_domain")
    email_local = _email_local(full_name)
    household_member = _pick(_CONTACT_NAME_POOL, seed + ":household")
    if household_member.split()[-1] == last_name:
        household_member = _pick(_CONTACT_NAME_POOL, seed + ":household_alt")
    close_contacts = tuple(
        name for name in _CONTACT_NAME_POOL if name != household_member
    )[:4]
    work_contacts = tuple(reversed(_CONTACT_NAME_POOL))[:4]
    return Persona(
        slug=slug,
        full_name=full_name,
        display_name=display_name,
        personal_email=f"{email_local}@{personal_domain}",
        work_email=f"{email_local}@{_company_domain(company)}",
        mobile_phone=_phone_number(city_profile["area_code"], seed + ":primary_phone"),
        home_address=_address(city_profile, seed),
        city=city_profile["city"],
        state=city_profile["state"],
        postal_code=_postal_code(city_profile, seed),
        timezone=city_profile["timezone"],
        employer_name=company,
        department=department,
        job_title=job_title,
        household_member_name=household_member,
        household_member_relationship=_pick(["spouse", "partner", "roommate", "sibling"], seed + ":relationship"),
        household_member_email=f"{_email_local(household_member)}@{_pick(_PERSONAL_EMAIL_DOMAINS, seed + ':household_domain')}",
        household_member_phone=_phone_number(city_profile["area_code"], seed + ":household_phone"),
        close_contacts=close_contacts,
        work_contacts=work_contacts,
    )


def _persona_context(persona: Persona | None) -> dict[str, Any]:
    if persona is None:
        return {}
    return {
        "primary_user": {
            "slug": persona.slug,
            "full_name": persona.full_name,
            "display_name": persona.display_name,
            "personal_email": persona.personal_email,
            "work_email": persona.work_email,
            "mobile_phone": persona.mobile_phone,
            "home_address": persona.home_address,
            "city": persona.city,
            "state": persona.state,
            "postal_code": persona.postal_code,
            "timezone": persona.timezone,
            "employer_name": persona.employer_name,
            "department": persona.department,
            "job_title": persona.job_title,
        },
        "household_context": {
            "member_name": persona.household_member_name,
            "relationship": persona.household_member_relationship,
            "email": persona.household_member_email,
            "phone": persona.household_member_phone,
        },
        "relationship_context": {
            "close_contacts": list(persona.close_contacts),
            "work_contacts": list(persona.work_contacts),
        },
        "consistency_contract": [
            "Use the primary_user as the owner of every app/account/export.",
            "Use household_context for household records instead of inventing a different spouse/roommate.",
            "Use relationship_context names as recurring contacts when contact or chat data needs people.",
            "Do not mention Alice Chen, Bob Smith, John Doe, Jane Doe, fake, sample, sandbox, benchmark, synthetic, OpenRouter, Kimi, or OpenAI in any JSON value.",
        ],
    }


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _resolve_structure_file(env_root: Path, explicit: str | None) -> Path:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    candidates.append(SCRIPT_DIR / "structure.md")
    candidates.append(env_root / "structure.md")

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()

    raise FileNotFoundError(
        "Could not find structure markdown. Tried: " + ", ".join(str(c) for c in candidates)
    )


def _parse_comment(comment: str) -> tuple[str, str]:
    comment = comment.strip()
    if not comment:
        return "snapshot", "medium"

    parts = [p.strip() for p in comment.split(",") if p.strip()]
    time_span = parts[0] if parts else "snapshot"
    sensitivity = "medium"
    for token in parts[1:] + parts[:1]:
        t = token.lower()
        if t in SENSITIVITY_LEVELS:
            sensitivity = t
            break
    return time_span, sensitivity


def parse_structure(markdown_text: str) -> list[FileSpec]:
    specs: list[FileSpec] = []
    in_tree = False
    stack: list[str] = []

    for raw_line in markdown_text.splitlines():
        line = raw_line.rstrip("\n")
        stripped = line.strip()

        if stripped.startswith("env_assets/"):
            in_tree = True
            stack = []
            continue

        if not in_tree:
            continue

        if stripped.startswith("## "):
            break

        if "|-" not in line:
            continue

        prefix, after = line.split("|-", 1)
        depth = prefix.count("|")
        entry = after.strip()

        comment = ""
        if "#" in entry:
            entry, comment = entry.split("#", 1)
            entry = entry.strip()

        if not entry:
            continue

        if entry.endswith("/"):
            folder = entry[:-1].strip()
            if len(stack) <= depth:
                stack.extend([""] * (depth - len(stack) + 1))
            stack[depth] = folder
            stack = stack[: depth + 1]
            continue

        if not entry.endswith(".json"):
            continue

        parents = [p for p in stack[:depth] if p]
        rel_path = "/".join(parents + [entry])
        time_span, sensitivity = _parse_comment(comment)
        specs.append(FileSpec(rel_path=rel_path, time_span=time_span, sensitivity=sensitivity))

    if not specs:
        raise ValueError("No JSON file specs parsed from structure markdown.")

    return specs


def _matches_prefix(rel_path: str, prefixes: tuple[str, ...]) -> bool:
    return any(rel_path.startswith(prefix) for prefix in prefixes)


def _structure_score(spec: FileSpec, seed: str) -> float:
    priority = _SENSITIVITY_PRIORITY.get(spec.sensitivity.lower(), 0.0)
    return _stable_float(f"{seed}:structure:{spec.rel_path}") + priority


def select_variable_structure(
    specs: list[FileSpec],
    seed: str,
    min_files: int = DEFAULT_MIN_STRUCTURE_FILES,
    max_files: int = DEFAULT_MAX_STRUCTURE_FILES,
) -> tuple[list[FileSpec], dict[str, Any]]:
    if not specs:
        return [], {"seed": seed, "selected_files": [], "omitted_files": [], "group_counts": {}}

    by_path = {spec.rel_path: spec for spec in specs}
    selected: dict[str, FileSpec] = {
        rel_path: by_path[rel_path]
        for rel_path in _REQUIRED_STRUCTURE_PATHS
        if rel_path in by_path
    }
    group_counts: dict[str, int] = {}

    for group_name, prefixes, min_count in _STRUCTURE_GROUP_RULES:
        group_specs = [spec for spec in specs if _matches_prefix(spec.rel_path, prefixes)]
        already_selected = [spec for spec in group_specs if spec.rel_path in selected]
        needed = max(0, min_count - len(already_selected))
        ranked = sorted(
            (spec for spec in group_specs if spec.rel_path not in selected),
            key=lambda spec: (_structure_score(spec, f"{seed}:{group_name}"), spec.rel_path),
        )
        for spec in ranked[:needed]:
            selected[spec.rel_path] = spec
        group_counts[group_name] = sum(1 for spec in group_specs if spec.rel_path in selected)

    total = len(specs)
    min_bound = min(total, max(1, min_files))
    max_bound = min(total, max(min_bound, max_files))
    if min_bound == max_bound:
        target_count = min_bound
    else:
        target_count = min_bound + (_stable_int(f"{seed}:target_count") % (max_bound - min_bound + 1))

    remaining = sorted(
        (spec for spec in specs if spec.rel_path not in selected),
        key=lambda spec: (_structure_score(spec, seed), spec.rel_path),
    )
    for spec in remaining:
        if len(selected) >= target_count:
            break
        selected[spec.rel_path] = spec

    order_index = {spec.rel_path: idx for idx, spec in enumerate(specs)}
    selected_specs = sorted(selected.values(), key=lambda spec: order_index[spec.rel_path])
    selected_paths = [spec.rel_path for spec in selected_specs]
    omitted_paths = [spec.rel_path for spec in specs if spec.rel_path not in selected]

    final_group_counts = {
        group_name: sum(1 for rel_path in selected_paths if _matches_prefix(rel_path, prefixes))
        for group_name, prefixes, _ in _STRUCTURE_GROUP_RULES
    }
    top_level_counts: dict[str, int] = {}
    for rel_path in selected_paths:
        top_level = rel_path.split("/", 1)[0]
        top_level_counts[top_level] = top_level_counts.get(top_level, 0) + 1

    plan = {
        "mode": "variable_structure",
        "seed": seed,
        "target_count": target_count,
        "available_count": total,
        "selected_count": len(selected_paths),
        "omitted_count": len(omitted_paths),
        "selected_files": selected_paths,
        "omitted_files": omitted_paths,
        "group_counts": final_group_counts,
        "top_level_counts": dict(sorted(top_level_counts.items())),
        "required_files_present": sorted(path for path in _REQUIRED_STRUCTURE_PATHS if path in by_path),
    }
    return selected_specs, plan


def _record_count_for_span(time_span: str) -> int:
    t = time_span.lower().strip()
    if "snapshot" in t:
        return 6
    if "7d" in t:
        return 8
    if "30d" in t:
        return 12
    if "90d" in t:
        return 20
    if "180d" in t:
        return 28
    if "365d" in t:
        return 36
    if "12m" in t:
        return 36
    if "24m" in t:
        return 48
    if "36m" in t:
        return 54
    if "84m" in t:
        return 60
    return 24


def _safe_output_path(output_root: Path, rel_path: str) -> Path:
    target = (output_root / rel_path).resolve()
    root = output_root.resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Unsafe output path escaped root: {rel_path}") from exc
    return target


def _build_seed_context(env_root: Path, persona: Persona | None = None) -> dict[str, Any]:
    persona_payload = _persona_context(persona)
    context: dict[str, Any] = {
        "user_name_hint": persona.display_name if persona else "Alice",
        "known_seed_files": [],
        "timeline_hint": {},
        **persona_payload,
    }

    seed_files = [
        "calendar/working/events_past_12m_future_6m.json",
        "email/inbox/threads_180d.json",
        "finance/banking/transactions_24m.json",
        "health/fitness/workouts_12m.json",
        "notes/knowledge/meeting_notes_365d.json",
    ]

    for rel_path in seed_files:
        path = env_root / rel_path
        if not path.exists():
            continue
        context["known_seed_files"].append(rel_path)
        try:
            data = json.loads(_read_text(path))
        except Exception:
            continue

        records = data.get("records", []) if isinstance(data, dict) else []

        if rel_path == "calendar/working/events_past_12m_future_6m.json":
            events = records if isinstance(records, list) else []
            starts = [e.get("start_time", "") for e in events if isinstance(e, dict)]
            starts = [s for s in starts if s]
            if starts:
                context["timeline_hint"]["calendar_start_min"] = min(starts)
                context["timeline_hint"]["calendar_start_max"] = max(starts)

        if rel_path == "finance/banking/transactions_24m.json":
            txs = records if isinstance(records, list) else []
            dates = [t.get("date", "") for t in txs if isinstance(t, dict)]
            dates = [d for d in dates if d]
            if dates:
                context["timeline_hint"]["bank_date_min"] = min(dates)
                context["timeline_hint"]["bank_date_max"] = max(dates)

    return context


def _strip_code_fence(text: str) -> str:
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9_-]*\n", "", s)
        if s.endswith("```"):
            s = s[:-3]
    return s.strip()


def _extract_json_object_text(text: str) -> str:
    """Try to extract the outermost JSON object from mixed text."""
    s = text.strip()
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return s
    return s[start : end + 1]


def _extract_text_from_sse_stream(raw: str) -> str:
    """Parse SSE text payload and reconstruct assistant content from chunk deltas."""
    content_parts: list[str] = []
    message_parts: list[str] = []

    for line in raw.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue

        payload = line[len("data:") :].strip()
        if not payload or payload == "[DONE]":
            continue

        try:
            evt = json.loads(payload)
        except json.JSONDecodeError:
            continue

        if not isinstance(evt, dict):
            continue

        choices = evt.get("choices")
        if isinstance(choices, list):
            for choice in choices:
                if not isinstance(choice, dict):
                    continue

                delta = choice.get("delta")
                if isinstance(delta, dict):
                    piece = delta.get("content")
                    if isinstance(piece, str):
                        content_parts.append(piece)

                msg = choice.get("message")
                if isinstance(msg, dict):
                    msg_content = msg.get("content")
                    if isinstance(msg_content, str):
                        message_parts.append(msg_content)

        if isinstance(evt.get("content"), str):
            message_parts.append(evt["content"])

    text = "".join(content_parts).strip()
    if text:
        return text
    return "\n".join(part for part in message_parts if part).strip()


def _has_time_field(record: dict[str, Any]) -> bool:
    time_keys = {"timestamp", "date", "start_time", "end_time", "period_start", "period_end"}
    return any(k in record for k in time_keys)


def _contains_disallowed_secret(raw_text: str) -> bool:
    patterns = [
        r'"(?:password|raw_password|plain_password)"\s*:',
        r"\brecovery code\b",
        r"\bseed phrase\b",
        r"\bprivate key\b",
        r"\bsocial security\b",
        r"\bssn\b",
        r"\bcvv\b",
    ]
    return any(re.search(pattern, raw_text, flags=re.IGNORECASE) for pattern in patterns)


def _find_benchmark_artifacts(raw_text: str) -> list[str]:
    hits: list[str] = []
    for pattern, label in _BENCHMARK_ARTIFACT_PATTERNS:
        if re.search(pattern, raw_text, flags=re.IGNORECASE):
            hits.append(label)
    return hits


def _source_for_path(rel_path: str) -> str:
    top_level = rel_path.split("/", 1)[0]
    second_level = rel_path.split("/", 2)[1] if "/" in rel_path else top_level
    mapping = {
        "agent_runtime_seed": "assistant_settings_export",
        "browser": "browser_history_export",
        "calendar": "calendar_sync",
        "desktop_apps": "desktop_app_export",
        "email": "mail_export",
        "files": "file_index_export",
        "finance": "finance_app_export",
        "health": "health_app_export",
        "home_life": "home_manager_export",
        "messaging": "chat_export",
        "meta": "account_settings_export",
        "notes": "notes_export",
        "notifications": "notification_center_export",
        "people": "contacts_export",
        "security": "security_center_export",
        "tasks": "task_manager_export",
        "travel": "travel_app_export",
    }
    if top_level == "desktop_apps":
        return f"{second_level}_export"
    return mapping.get(top_level, "local_app_export")


def _normalize_metadata(obj: dict[str, Any], spec: FileSpec) -> dict[str, Any]:
    meta = obj.get("_meta")
    if not isinstance(meta, dict):
        return obj

    for key in list(meta.keys()):
        if key.lower() in _BLOCKED_META_KEYS:
            meta.pop(key, None)

    source = str(meta.get("source", "")).strip()
    if not source or _find_benchmark_artifacts(source):
        meta["source"] = _source_for_path(spec.rel_path)

    return obj


def _validate_persona_consistency(spec: FileSpec, obj: dict[str, Any], persona: Persona | None) -> None:
    if persona is None:
        return

    normalized_text = json.dumps(obj, ensure_ascii=False).lower()
    blocked_defaults = [
        "alice chen",
        "alice marie chen",
        "bob smith",
        "4527 birchwood",
        "123 maple street",
    ]
    for blocked in blocked_defaults:
        if blocked in normalized_text and blocked not in persona.full_name.lower():
            raise ValidationError(f"Output leaked default seed identity/detail: {blocked}")

    if spec.rel_path == "meta/user_profile.json":
        required_values = [
            persona.full_name,
            persona.personal_email,
            persona.work_email,
            persona.mobile_phone,
            persona.city,
            persona.employer_name,
        ]
        missing = [value for value in required_values if value.lower() not in normalized_text]
        if missing:
            raise ValidationError("user_profile missing persona fields: " + ", ".join(missing[:3]))

    if spec.rel_path == "meta/household_profile.json":
        required_values = [persona.full_name, persona.household_member_name, persona.city]
        missing = [value for value in required_values if value.lower() not in normalized_text]
        if missing:
            raise ValidationError("household_profile missing persona fields: " + ", ".join(missing[:3]))


def _extract_text_from_content_field(content: Any) -> str:
    """Normalize possible content shapes into a plain text string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if isinstance(item.get("text"), str):
                    parts.append(item["text"])
                elif item.get("type") == "output_text" and isinstance(item.get("content"), str):
                    parts.append(item["content"])
            elif hasattr(item, "text") and isinstance(getattr(item, "text"), str):
                parts.append(getattr(item, "text"))
            elif hasattr(item, "content") and isinstance(getattr(item, "content"), str):
                parts.append(getattr(item, "content"))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(p for p in parts if p).strip()
    if hasattr(content, "text") and isinstance(getattr(content, "text"), str):
        return getattr(content, "text")
    return ""


def _extract_response_text(response: Any) -> str:
    """
    Extract assistant text from multiple OpenAI-compatible response shapes.

    Supports:
    - raw string responses
    - dict-like responses
    - OpenAI ChatCompletion objects with .choices[0].message.content
    - responses-style objects with .output_text
    """
    if isinstance(response, str):
        # Some OpenAI-compatible providers return SSE chunks as one raw string.
        if "data:" in response:
            parsed = _extract_text_from_sse_stream(response)
            if parsed:
                return parsed
        return response

    model_dump = getattr(response, "model_dump", None)
    if callable(model_dump):
        try:
            dumped = model_dump()
            text = _extract_response_text(dumped)
            if text.strip():
                return text
        except Exception:
            pass

    if isinstance(response, dict):
        choices = response.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                msg = first.get("message", {})
                if isinstance(msg, dict):
                    return _extract_text_from_content_field(msg.get("content"))
        if isinstance(response.get("output_text"), str):
            return response["output_text"]
        if isinstance(response.get("content"), str):
            return response["content"]
        return json.dumps(response, ensure_ascii=False)

    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text

    output = getattr(response, "output", None)
    if isinstance(output, list):
        chunks: list[str] = []
        for block in output:
            content = getattr(block, "content", None)
            text = _extract_text_from_content_field(content)
            if text:
                chunks.append(text)
        if chunks:
            return "\n".join(chunks).strip()

    choices = getattr(response, "choices", None)
    if choices:
        first = choices[0]
        msg = getattr(first, "message", None)
        if msg is not None:
            content = getattr(msg, "content", None)
            text = _extract_text_from_content_field(content)
            if text:
                return text
        # Some providers return text directly in choice.text
        choice_text = getattr(first, "text", None)
        if isinstance(choice_text, str) and choice_text.strip():
            return choice_text

    return str(response)


def validate_output(
    spec: FileSpec,
    output_text: str,
    expected_count: int,
    persona: Persona | None = None,
) -> dict[str, Any]:
    cleaned = _strip_code_fence(output_text)
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        # Fallback: many providers prepend prose around JSON.
        extracted = _extract_json_object_text(cleaned)
        try:
            obj = json.loads(extracted)
        except json.JSONDecodeError:
            raise ValidationError(f"Invalid JSON: {exc}") from exc

    if not isinstance(obj, dict):
        raise ValidationError("Top-level JSON must be an object.")

    obj = _normalize_metadata(obj, spec)

    required_top = {"_meta", "records"}
    if set(obj.keys()) != required_top:
        raise ValidationError("Top-level keys must be exactly: _meta, records.")

    meta = obj.get("_meta")
    records = obj.get("records")

    if not isinstance(meta, dict):
        raise ValidationError("_meta must be an object.")
    if not isinstance(records, list):
        raise ValidationError("records must be an array.")

    for key in ["schema_version", "file_path", "time_span", "sensitivity", "generated_at_utc", "source"]:
        if key not in meta:
            raise ValidationError(f"_meta missing required key: {key}")

    blocked_meta = [key for key in meta if key.lower() in _BLOCKED_META_KEYS]
    if blocked_meta:
        raise ValidationError("_meta contains benchmark-revealing keys: " + ", ".join(blocked_meta[:4]))

    if meta["file_path"] != spec.rel_path:
        raise ValidationError("_meta.file_path does not match target file path.")
    if str(meta["time_span"]).strip().lower() != spec.time_span.strip().lower():
        raise ValidationError("_meta.time_span does not match target time span.")
    if str(meta["sensitivity"]).strip().lower() != spec.sensitivity.strip().lower():
        raise ValidationError("_meta.sensitivity does not match target sensitivity.")
    if meta["schema_version"] != "1.0":
        raise ValidationError("_meta.schema_version must be '1.0'.")
    if not isinstance(meta.get("source"), str) or not meta["source"].strip():
        raise ValidationError("_meta.source must be a non-empty string.")

    if len(records) != expected_count:
        raise ValidationError(f"records must contain exactly {expected_count} items.")

    for idx, rec in enumerate(records):
        if not isinstance(rec, dict):
            raise ValidationError(f"records[{idx}] must be an object.")
        if "id" not in rec or not isinstance(rec["id"], str) or not rec["id"].strip():
            raise ValidationError(f"records[{idx}] must contain non-empty string field 'id'.")
        if not _has_time_field(rec):
            raise ValidationError(f"records[{idx}] must include at least one time field.")

    normalized_text = json.dumps(obj, ensure_ascii=False)

    if spec.sensitivity != "critical" and _contains_disallowed_secret(normalized_text):
        raise ValidationError("Non-critical file contains disallowed secret-like fields.")

    benchmark_artifacts = _find_benchmark_artifacts(normalized_text)
    if benchmark_artifacts:
        detail = ", ".join(benchmark_artifacts[:4])
        raise ValidationError(
            "Output contains benchmark-revealing placeholders: " + detail
        )

    _validate_persona_consistency(spec, obj, persona)

    return obj


def generate_one_file(
    client: OpenAI,
    model: str,
    spec: FileSpec,
    seed_context: dict[str, Any],
    max_retries: int,
    persona: Persona | None = None,
) -> dict[str, Any]:
    target_count = _record_count_for_span(spec.time_span)

    user_payload = {
        "task": "Generate one JSON document for the requested path.",
        "target": {
            "file_path": spec.rel_path,
            "time_span": spec.time_span,
            "sensitivity": spec.sensitivity,
            "target_record_count": target_count,
        },
        "seed_context": seed_context,
        "quality_checks": [
            "The JSON must pass a post-generation audit for placeholder names, 555 numbers, fake IDs, sandbox/benchmark/synthetic markers, and provider/model markers.",
            "The JSON must remain consistent with seed_context.primary_user when one is provided.",
        ],
    }

    last_error = ""
    for attempt in range(1, max_retries + 1):
        messages: list[dict[str, str]] = [
            {"role": "system", "content": GENERATION_RULES},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ]

        if last_error:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Previous output failed validation. Regenerate from scratch and fix these issues exactly:\n"
                        + last_error
                    ),
                }
            )

        response = client.chat.completions.create(model=model, messages=messages, stream=False)
        content = _extract_response_text(response)

        if not isinstance(content, str) or not content.strip():
            last_error = "Empty or unsupported response payload from model provider."
            print(f"  - retry {attempt}/{max_retries} failed: {last_error}")
            continue

        try:
            return validate_output(spec, content, target_count, persona=persona)
        except ValidationError as exc:
            last_error = str(exc)
            print(f"  - retry {attempt}/{max_retries} failed: {last_error}")

    raise ValidationError(f"Failed after {max_retries} attempts: {last_error}")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def append_run_log(log_path: Path, event: str, **payload: Any) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    record = {"timestamp": _utc_now(), "event": event, **payload}
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    write_json(path, payload)


def prune_unselected_files(env_root: Path, selected_rel_paths: set[str]) -> list[str]:
    if not env_root.exists():
        return []
    pruned: list[str] = []
    for path in sorted(env_root.rglob("*.json")):
        if not path.is_file():
            continue
        rel_path = path.relative_to(env_root).as_posix()
        if rel_path in selected_rel_paths:
            continue
        path.unlink()
        pruned.append(rel_path)
    return pruned


def existing_file_status(path: Path, spec: FileSpec, persona: Persona | None) -> tuple[bool, str]:
    if not path.exists():
        return False, "missing"
    try:
        validate_output(
            spec=spec,
            output_text=_read_text(path),
            expected_count=_record_count_for_span(spec.time_span),
            persona=persona,
        )
    except Exception as exc:
        return False, str(exc)
    return True, "valid"


def audit_written_files(env_root: Path, specs: list[FileSpec], persona: Persona | None) -> list[str]:
    errors: list[str] = []
    for spec in specs:
        path = _safe_output_path(env_root, spec.rel_path)
        if not path.exists():
            errors.append(f"missing: {spec.rel_path}")
            continue
        try:
            validate_output(
                spec=spec,
                output_text=_read_text(path),
                expected_count=_record_count_for_span(spec.time_span),
                persona=persona,
            )
        except Exception as exc:
            errors.append(f"{spec.rel_path}: {exc}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate an env_assets-compatible JSON tree, one file at a time with validation."
    )
    parser.add_argument(
        "--env-root",
        default=None,
        help=(
            "Root output directory. Defaults to repo_root/env_assets, or "
            "<utility-root>/<persona-slug> when --persona-name/--persona-slug is provided."
        ),
    )
    parser.add_argument(
        "--utility-root",
        default=str(DEFAULT_UTILITY_ROOT),
        help="Parent directory for persona utility-test envs (default: repo_root/env_assets_utility_test)",
    )
    parser.add_argument("--persona-name", default="", help="Primary user full name for a utility-test environment")
    parser.add_argument("--persona-slug", default="", help="Folder slug for the utility-test user")
    parser.add_argument("--persona-seed", default="", help="Optional deterministic seed for persona details")
    parser.add_argument("--print-persona", action="store_true", help="Print the derived persona context and exit")
    parser.add_argument("--plan-only", action="store_true", help="Print target root and file plan without calling the model")
    parser.add_argument("--structure-file", default=None, help="Path to structure markdown")
    parser.add_argument("--variable-structure", action="store_true", help="Select a rich seed-driven subset of structure.md instead of generating every file")
    parser.add_argument("--structure-seed", default="", help="Seed controlling which files are selected when --variable-structure is used")
    parser.add_argument("--min-structure-files", type=int, default=DEFAULT_MIN_STRUCTURE_FILES, help="Minimum files for --variable-structure")
    parser.add_argument("--max-structure-files", type=int, default=DEFAULT_MAX_STRUCTURE_FILES, help="Maximum files for --variable-structure")
    parser.add_argument(
        "--mirror-env-root",
        default="",
        help=(
            "Optional existing env root whose JSON path set should be mirrored. "
            "Use this to make utility-test envs exactly match the current env_assets tree."
        ),
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="OpenAI-compatible base URL")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model name")
    parser.add_argument("--api-key-env", default=DEFAULT_API_KEY_ENV, help="API key env var name")
    parser.add_argument("--max-retries", type=int, default=3, help="Retries per file on validation failure")
    parser.add_argument("--max-files", type=int, default=0, help="Generate only first N files (0 = all)")
    parser.add_argument("--only", default="", help="Generate only paths containing this substring")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")
    parser.add_argument("--resume", action="store_true", help="Skip existing files that pass validation and regenerate missing/invalid files")
    parser.add_argument("--log-root", default=str(REPO_ROOT / "prompt_optimization" / "logs" / "env_generation"), help="Directory for generator manifests and JSONL run logs")
    parser.add_argument("--run-name", default="", help="Optional name for this run under --log-root")
    parser.add_argument(
        "--prune-unselected",
        action="store_true",
        help="With --variable-structure, remove existing JSON files under env-root that are not selected by this run",
    )
    args = parser.parse_args()

    persona_name = args.persona_name.strip()
    if not persona_name and args.persona_slug.strip():
        persona_name = args.persona_slug.replace("-", " ").replace("_", " ").title()
    persona = (
        build_persona(persona_name, args.persona_slug, args.persona_seed)
        if persona_name or args.persona_slug.strip()
        else None
    )

    if args.env_root:
        env_root = Path(args.env_root).expanduser().resolve()
    elif persona is not None:
        env_root = (Path(args.utility_root).expanduser() / persona.slug).resolve()
    else:
        env_root = DEFAULT_ENV_ROOT.resolve()

    structure_path = _resolve_structure_file(env_root, args.structure_file)

    if args.print_persona:
        if persona is None:
            raise SystemExit("--print-persona requires --persona-name or --persona-slug")
        print(json.dumps(_persona_context(persona), ensure_ascii=False, indent=2))
        print(f"target_env_root: {env_root}")
        return

    structure_text = _read_text(structure_path)
    specs = parse_structure(structure_text)
    structure_seed = (
        args.structure_seed.strip()
        or args.persona_seed.strip()
        or (persona.slug if persona is not None else env_root.name)
    )
    run_name = args.run_name.strip() or (persona.slug if persona is not None else _slugify(env_root.name))
    log_dir = (Path(args.log_root).expanduser() / run_name).resolve()
    manifest_path = log_dir / "manifest.json"
    run_log_path = log_dir / "generation.jsonl"

    mirror_env_root = Path(args.mirror_env_root).expanduser().resolve() if args.mirror_env_root else None
    if mirror_env_root is not None:
        if not mirror_env_root.exists() or not mirror_env_root.is_dir():
            raise SystemExit(f"--mirror-env-root is not a directory: {mirror_env_root}")
        mirror_paths = {
            path.relative_to(mirror_env_root).as_posix()
            for path in mirror_env_root.rglob("*.json")
            if path.is_file()
        }
        known_paths = {spec.rel_path for spec in specs}
        unknown_paths = sorted(mirror_paths - known_paths)
        if unknown_paths:
            print("Warning: mirror env has JSON files not listed in structure.md; ignoring:")
            for rel_path in unknown_paths[:10]:
                print(f"  - {rel_path}")
            if len(unknown_paths) > 10:
                print(f"  ... {len(unknown_paths) - 10} more")
        specs = [spec for spec in specs if spec.rel_path in mirror_paths]

    if args.variable_structure:
        if args.min_structure_files > args.max_structure_files:
            raise SystemExit("--min-structure-files cannot exceed --max-structure-files")
        specs, structure_plan = select_variable_structure(
            specs=specs,
            seed=structure_seed,
            min_files=max(1, args.min_structure_files),
            max_files=max(1, args.max_structure_files),
        )
    else:
        selected_paths = [spec.rel_path for spec in specs]
        structure_plan = {
            "mode": "full_structure",
            "seed": structure_seed,
            "available_count": len(specs),
            "selected_count": len(specs),
            "omitted_count": 0,
            "selected_files": selected_paths,
            "omitted_files": [],
            "top_level_counts": {
                top_level: sum(1 for rel_path in selected_paths if rel_path.split("/", 1)[0] == top_level)
                for top_level in sorted({rel_path.split("/", 1)[0] for rel_path in selected_paths})
            },
        }

    if mirror_env_root is not None:
        structure_plan["mirror_env_root"] = str(mirror_env_root)

    if args.only:
        specs = [s for s in specs if args.only in s.rel_path]

    if args.max_files > 0:
        specs = specs[: args.max_files]

    if not specs:
        raise SystemExit("No target files to generate after filtering.")

    planned_files = [spec.rel_path for spec in specs]
    generation_plan = {
        "created_at_utc": _utc_now(),
        "run_name": run_name,
        "env_root": str(env_root),
        "structure_path": str(structure_path),
        "structure_seed": structure_seed,
        "variable_structure": bool(args.variable_structure),
        "planned_count": len(planned_files),
        "planned_files": planned_files,
        "debug_filters": {"only": args.only, "max_files": args.max_files},
        "structure_plan": structure_plan,
        "persona": _persona_context(persona).get("primary_user") if persona is not None else None,
    }

    if args.prune_unselected and not args.variable_structure:
        raise SystemExit("--prune-unselected requires --variable-structure")
    if args.prune_unselected and env_root == DEFAULT_ENV_ROOT.resolve():
        raise SystemExit("Refusing to prune the default env_assets root")

    if args.plan_only:
        print(f"Structure file: {structure_path}")
        print(f"Target env root: {env_root}")
        print(f"Log dir: {log_dir}")
        if mirror_env_root is not None:
            print(f"Mirroring JSON path set from: {mirror_env_root}")
        if args.variable_structure:
            print(f"Variable structure seed: {structure_seed}")
            print(f"Selected/Omitted: {structure_plan['selected_count']}/{structure_plan['omitted_count']}")
            print("Top-level counts:")
            for key, value in structure_plan.get("top_level_counts", {}).items():
                print(f"  {key}: {value}")
        if persona is not None:
            print(f"Persona: {persona.full_name} ({persona.slug})")
        print(f"Files planned: {len(specs)}")
        for spec in specs[:10]:
            print(f"  - {spec.rel_path}")
        if len(specs) > 10:
            print(f"  ... {len(specs) - 10} more")
        return

    api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key:
        raise SystemExit(f"Missing API key env var: {args.api_key_env}")

    client = OpenAI(base_url=args.base_url, api_key=api_key)
    seed_context = _build_seed_context(env_root, persona=persona)
    seed_context["environment_structure"] = {
        "mode": structure_plan.get("mode"),
        "seed": structure_seed,
        "selected_files": planned_files,
        "omitted_files": structure_plan.get("omitted_files", []),
        "contract": [
            "Only refer to data sources that appear in selected_files as present in this user's environment.",
            "Do not invent app folders or JSON files that are listed in omitted_files.",
            "It is acceptable for this user environment to have a different app/file coverage pattern from other users.",
        ],
    }

    pruned_files: list[str] = []
    if args.prune_unselected:
        pruned_files = prune_unselected_files(env_root, set(planned_files))

    generation_plan.update({
        "model": args.model,
        "resume": bool(args.resume),
        "overwrite": bool(args.overwrite),
        "prune_unselected": bool(args.prune_unselected),
        "pruned_files": pruned_files,
        "max_retries": max(1, args.max_retries),
        "log_dir": str(log_dir),
        "run_log_path": str(run_log_path),
        "manifest_path": str(manifest_path),
    })
    write_manifest(manifest_path, generation_plan)
    append_run_log(run_log_path, "run_start", **generation_plan)
    if pruned_files:
        append_run_log(run_log_path, "pruned_unselected", count=len(pruned_files), files=pruned_files)

    print(f"Structure file: {structure_path}")
    print(f"Target env root: {env_root}")
    print(f"Log dir: {log_dir}")
    if mirror_env_root is not None:
        print(f"Mirroring JSON path set from: {mirror_env_root}")
    if args.variable_structure:
        print(f"Variable structure seed: {structure_seed}")
        print(f"Selected/Omitted: {structure_plan['selected_count']}/{structure_plan['omitted_count']}")
    if pruned_files:
        print(f"Pruned unselected JSON files: {len(pruned_files)}")
    if persona is not None:
        print(f"Persona: {persona.full_name} ({persona.slug})")
    print(f"Model: {args.model}")
    print(f"Files to generate: {len(specs)}")

    succeeded = 0
    skipped = 0
    failed = 0
    written_specs: list[FileSpec] = []
    resume_valid_specs: list[FileSpec] = []

    for idx, spec in enumerate(specs, start=1):
        out_path = _safe_output_path(env_root, spec.rel_path)
        if out_path.exists():
            if args.resume:
                valid, reason = existing_file_status(out_path, spec, persona)
                if valid:
                    print(f"[{idx}/{len(specs)}] resume skip valid: {spec.rel_path}")
                    append_run_log(
                        run_log_path,
                        "file_skipped",
                        rel_path=spec.rel_path,
                        reason="resume_valid",
                    )
                    resume_valid_specs.append(spec)
                    skipped += 1
                    continue
                print(f"[{idx}/{len(specs)}] resume regenerate invalid: {spec.rel_path} ({reason})")
                append_run_log(
                    run_log_path,
                    "file_regenerate_invalid",
                    rel_path=spec.rel_path,
                    reason=reason,
                )
            elif not args.overwrite:
                print(f"[{idx}/{len(specs)}] skip existing: {spec.rel_path}")
                append_run_log(
                    run_log_path,
                    "file_skipped",
                    rel_path=spec.rel_path,
                    reason="exists_without_overwrite",
                )
                skipped += 1
                continue

        print(f"[{idx}/{len(specs)}] generating: {spec.rel_path} ({spec.time_span}, {spec.sensitivity})")
        append_run_log(
            run_log_path,
            "file_start",
            rel_path=spec.rel_path,
            index=idx,
            total=len(specs),
            time_span=spec.time_span,
            sensitivity=spec.sensitivity,
        )
        try:
            payload = generate_one_file(
                client=client,
                model=args.model,
                spec=spec,
                seed_context=seed_context,
                max_retries=max(1, args.max_retries),
                persona=persona,
            )
            write_json(out_path, payload)
            append_run_log(run_log_path, "file_generated", rel_path=spec.rel_path)
            written_specs.append(spec)
            succeeded += 1
        except Exception as exc:
            failed += 1
            append_run_log(run_log_path, "file_failed", rel_path=spec.rel_path, error=str(exc))
            print(f"  - failed: {exc}")

    print("\nDone.")
    print(f"  succeeded: {succeeded}")
    print(f"  skipped:   {skipped}")
    print(f"  failed:    {failed}")

    if failed > 0:
        append_run_log(
            run_log_path,
            "run_end",
            status="failed",
            succeeded=succeeded,
            skipped=skipped,
            failed=failed,
        )
        raise SystemExit(1)

    if not written_specs:
        print("Final audit: skipped because no files were written")
        append_run_log(
            run_log_path,
            "run_end",
            status="success_no_writes",
            succeeded=succeeded,
            skipped=skipped,
            failed=failed,
            resume_valid=len(resume_valid_specs),
        )
        return

    audit_errors = audit_written_files(env_root, written_specs, persona)
    if audit_errors:
        print("\nFinal audit failed:")
        for error in audit_errors[:20]:
            print(f"  - {error}")
        if len(audit_errors) > 20:
            print(f"  ... {len(audit_errors) - 20} more")
        append_run_log(
            run_log_path,
            "run_end",
            status="audit_failed",
            succeeded=succeeded,
            skipped=skipped,
            failed=failed,
            audit_errors=audit_errors[:20],
        )
        raise SystemExit(1)

    print(f"Final audit: passed {len(written_specs)} written files")
    append_run_log(
        run_log_path,
        "run_end",
        status="success",
        succeeded=succeeded,
        skipped=skipped,
        failed=failed,
        audited=len(written_specs),
        resume_valid=len(resume_valid_specs),
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
