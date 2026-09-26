from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_provider_profile(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Provider profile must contain a mapping: {path}")
    return payload
