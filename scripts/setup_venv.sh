#!/usr/bin/env bash
# Create or refresh a local virtual environment for SEABench.
#
# Usage:
#   ./scripts/setup_venv.sh
#   ./scripts/setup_venv.sh .venv
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_DIR="${1:-$REPO_ROOT/.venv}"

echo "Creating virtual environment at $VENV_DIR"
python3 -m venv "$VENV_DIR"

source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip
pip install -r "$REPO_ROOT/requirements.txt"

echo
echo "Virtual environment is ready."
echo "Activate it with:"
echo "  source \"$VENV_DIR/bin/activate\""
