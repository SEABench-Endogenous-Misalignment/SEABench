#!/usr/bin/env python3
"""Compatibility shim for the renamed self-evolution attack-sequence generator."""

from __future__ import annotations

import runpy
from pathlib import Path


def main() -> None:
    target = Path(__file__).with_name("generate_self_evolution_attack_sequence.py")
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
