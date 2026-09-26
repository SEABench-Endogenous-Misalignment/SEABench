#!/usr/bin/env bash
# Run the personal assistant sandbox inside an Apptainer container.
#
# Prerequisites:
#   - Apptainer 1.3+ available (module load apptainer if needed)
#   - An API key environment variable set (defaults to OPENROUTER_API_KEY)
#   - Must be on an HPC compute node (not a login node)
#
# Usage:
#   export OPENROUTER_API_KEY="your-key-here"   # default provider profile: openrouter
#   ./scripts/run.sh
#   ./scripts/run.sh --run-config configs/runs/controller_update_computer_use_privacy.yaml
#   ./scripts/run.sh --run-config ... --resume-latest
#   ./scripts/run.sh --run-config ... --resume-run-dir artifacts/<run_id>
set -euo pipefail

module load apptainer

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
CONTAINER_DIR="$REPO_ROOT/sandbox"
SIF="$CONTAINER_DIR/sandbox.sif"

# ── Resolve provider + required API key env var (default: openrouter/OPENROUTER_API_KEY)
DEFAULT_RUN_CONFIG="$REPO_ROOT/configs/runs/controller_update_computer_use_privacy.yaml"
RUN_CONFIG_HOST="$DEFAULT_RUN_CONFIG"
for ((i = 1; i <= $#; i++)); do
  if [[ "${!i}" == "--run-config" ]]; then
    j=$((i + 1))
    if [[ $j -le $# ]]; then
      RUN_CONFIG_HOST="${!j}"
    fi
    break
  fi
done

# Allow passing in-container path (/repo/...) for convenience.
if [[ "$RUN_CONFIG_HOST" == /repo/* ]]; then
  RUN_CONFIG_HOST="$REPO_ROOT/${RUN_CONFIG_HOST#/repo/}"
elif [[ "$RUN_CONFIG_HOST" != /* ]]; then
  RUN_CONFIG_HOST="$REPO_ROOT/$RUN_CONFIG_HOST"
fi

if [[ ! -f "$RUN_CONFIG_HOST" ]]; then
  echo "ERROR: Run config not found: $RUN_CONFIG_HOST" >&2
  exit 1
fi

PROVIDER_PROFILE="$(grep -E '^[[:space:]]*profile:[[:space:]]*' "$RUN_CONFIG_HOST" | head -n 1 | sed -E 's/^[[:space:]]*profile:[[:space:]]*//')"
if [[ "$PROVIDER_PROFILE" == "" ]]; then
  PROVIDER_PROFILE="openrouter"
fi

PROVIDER_YAML="$REPO_ROOT/configs/providers/${PROVIDER_PROFILE}.yaml"
API_KEY_ENV="OPENROUTER_API_KEY"
if [[ -f "$PROVIDER_YAML" ]]; then
  API_KEY_ENV_LINE="$(grep -E '^[[:space:]]*api_key_env:[[:space:]]*' "$PROVIDER_YAML" | head -n 1 || true)"
  if [[ "$API_KEY_ENV_LINE" != "" ]]; then
    API_KEY_ENV="$(echo "$API_KEY_ENV_LINE" | sed -E 's/^[[:space:]]*api_key_env:[[:space:]]*//')"
  fi
fi

# ── Validate prerequisites ────────────────────────────────────────────────────
# If the selected provider profile expects a key env var, require it on the host.
if [[ "$API_KEY_ENV" != "" ]]; then
  if [[ -z "${!API_KEY_ENV:-}" ]]; then
      echo "ERROR: $API_KEY_ENV is not set (provider profile: $PROVIDER_PROFILE)." >&2
      echo "  export $API_KEY_ENV=\"your-key-here\"" >&2
      exit 1
  fi
fi

# ── Build the SIF image if not already present ────────────────────────────────
if [[ ! -f "$SIF" ]]; then
    echo "Building Apptainer image — this takes about a minute..."
    apptainer build "$SIF" "$CONTAINER_DIR/sandbox.def"
    echo "Image built: $SIF"
fi

# ── Launch ────────────────────────────────────────────────────────────────────
echo "Starting sandbox run..."
ARTIFACTS_DIR_IN_CONTAINER="${ARTIFACTS_DIR:-/repo/artifacts}"
apptainer exec \
    --bind "$REPO_ROOT":/repo \
    --bind "$REPO_ROOT/env_assets":/env_assets \
    --env "$API_KEY_ENV"="${!API_KEY_ENV:-}" \
    --env PYTHONPATH=/repo \
    --env REPO_ROOT=/repo \
    --env ARTIFACTS_DIR="$ARTIFACTS_DIR_IN_CONTAINER" \
    --env ENV_ASSETS_DIR=/env_assets \
    "$SIF" \
    python3 /repo/scripts/run.py "$@"
