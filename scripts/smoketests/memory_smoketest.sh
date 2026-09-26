#!/usr/bin/env bash
# Memory smoketest — runs memory_smoketest.py inside the Apptainer container.
#
# Requires OPENROUTER_API_KEY to be set (live API call).
#
# Usage:
#   export OPENROUTER_API_KEY="your-key"
#   ./scripts/smoketests/memory_smoketest.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONTAINER_DIR="$REPO_ROOT/sandbox"
SIF="$CONTAINER_DIR/sandbox.sif"

if [[ ! -f "$SIF" ]]; then
    echo "Building Apptainer image — this takes about a minute..."
    apptainer build "$SIF" "$CONTAINER_DIR/sandbox.def"
fi

echo "Running memory smoketest..."
apptainer exec \
    --bind "$REPO_ROOT":/repo \
    --bind "$REPO_ROOT/env_assets":/env_assets \
    --env OPENROUTER_API_KEY="${OPENROUTER_API_KEY:-}" \
    --env PYTHONPATH=/repo \
    --env REPO_ROOT=/repo \
    --env ARTIFACTS_DIR=/repo/artifacts \
    --env ENV_ASSETS_DIR=/env_assets \
    "$SIF" \
    python3 /repo/scripts/smoketests/memory_smoketest.py
