#!/usr/bin/env bash
# Smoke test — spins up the Apptainer container and runs test_smoke.py
#
# Steps 1-4 run without an API key (pure Python).
# Step 5 (live agent session) runs only if OPENROUTER_API_KEY is set.
#
# Usage:
#   ./scripts/test_smoke.sh                # local checks only
#   export OPENROUTER_API_KEY="your-key"
#   ./scripts/test_smoke.sh                # includes live API call
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
CONTAINER_DIR="$REPO_ROOT/sandbox"
SIF="$CONTAINER_DIR/sandbox.sif"

if [[ ! -f "$SIF" ]]; then
    echo "Building Apptainer image — this takes about a minute..."
    apptainer build "$SIF" "$CONTAINER_DIR/sandbox.def"
fi

echo "Running smoke test..."
apptainer exec \
    --bind "$REPO_ROOT":/repo \
    --bind "$REPO_ROOT/env_assets":/env_assets \
    --env OPENROUTER_API_KEY="${OPENROUTER_API_KEY:-}" \
    --env PYTHONPATH=/repo \
    --env REPO_ROOT=/repo \
    --env ARTIFACTS_DIR=/repo/artifacts \
    --env ENV_ASSETS_DIR=/env_assets \
    "$SIF" \
    python3 /repo/scripts/test_smoke.py
