#!/usr/bin/env bash
# Run one-shot A0 vs evolved utility tests across every alternate-user environment.

set -uo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash prompt_optimization/run_counterfactual_utility_sweep.sh \
    --log-dir PATH [--output-dir PATH] [options]

Required:
  --log-dir PATH                  Completed refinement trajectory.

Options:
  --output-dir PATH               Default: <log-dir>/counterfactual_utility
  --provider-profile PROFILE      Agent/adaptation provider (default: openrouter)
  --agent-model MODEL             Agent and task-adaptation model override
  --judge-provider-profile NAME   Utility judge provider (default: openrouter)
  --judge-model MODEL             Utility judge model override
  --max-task-retries N            Per-task retries (default: 0; one-shot)
  --max-attempts N                Whole-sweep process attempts (default: 20)
  --sleep-seconds N               Delay between process attempts (default: 5)
  --resume                        Resume an existing matching output directory
  --dry-run                       Validate and print all planned runs; no API calls
  --prepare-only                  Adapt tasks for review; do not run utility tasks
  -h, --help                      Show this help

The sweep always evaluates all five directories under env_assets_utility_test/.
EOF
}

LOG_DIR=""
OUTPUT_DIR=""
PROVIDER_PROFILE="openrouter"
AGENT_MODEL=""
JUDGE_PROVIDER_PROFILE="openrouter"
JUDGE_MODEL=""
MAX_TASK_RETRIES=0
MAX_ATTEMPTS=20
SLEEP_SECONDS=5
RESUME=false
DRY_RUN=false
PREPARE_ONLY=false

while [ "$#" -gt 0 ]; do
    case "$1" in
        --log-dir|--output-dir|--provider-profile|--agent-model|--judge-provider-profile|--judge-model|--max-task-retries|--max-attempts|--sleep-seconds)
            [ "$#" -ge 2 ] || { echo "ERROR: $1 requires a value" >&2; exit 2; }
            case "$1" in
                --log-dir) LOG_DIR=$2 ;;
                --output-dir) OUTPUT_DIR=$2 ;;
                --provider-profile) PROVIDER_PROFILE=$2 ;;
                --agent-model) AGENT_MODEL=$2 ;;
                --judge-provider-profile) JUDGE_PROVIDER_PROFILE=$2 ;;
                --judge-model) JUDGE_MODEL=$2 ;;
                --max-task-retries) MAX_TASK_RETRIES=$2 ;;
                --max-attempts) MAX_ATTEMPTS=$2 ;;
                --sleep-seconds) SLEEP_SECONDS=$2 ;;
            esac
            shift 2
            ;;
        --resume)
            RESUME=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --prepare-only)
            PREPARE_ONLY=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

[ -n "$LOG_DIR" ] || { echo "ERROR: --log-dir is required" >&2; exit 2; }
case "$MAX_ATTEMPTS:$SLEEP_SECONDS:$MAX_TASK_RETRIES" in
    *[!0-9:]*|:*|*::*|*:)
        echo "ERROR: retry values must be non-negative integers" >&2
        exit 2
        ;;
esac
[ "$MAX_ATTEMPTS" -ge 1 ] || { echo "ERROR: --max-attempts must be at least 1" >&2; exit 2; }

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
PYTHON=${PYTHON:-python}
EVALUATOR="$SCRIPT_DIR/evaluate_utility_gain.py"

if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="${LOG_DIR%/}/counterfactual_utility"
fi

base_command=(
    "$PYTHON" -u "$EVALUATOR"
    --log-dir "$LOG_DIR"
    --output-dir "$OUTPUT_DIR"
    --users all
    --provider-profile "$PROVIDER_PROFILE"
    --judge-provider-profile "$JUDGE_PROVIDER_PROFILE"
    --max-task-retries "$MAX_TASK_RETRIES"
)
[ -n "$AGENT_MODEL" ] && base_command+=(--agent-model "$AGENT_MODEL")
[ -n "$JUDGE_MODEL" ] && base_command+=(--judge-model "$JUDGE_MODEL")
[ "$PREPARE_ONLY" = true ] && base_command+=(--prepare-only)

if [ "$DRY_RUN" = true ]; then
    echo "Validating counterfactual utility sweep (no API calls) ..."
    exec "${base_command[@]}" --dry-run
fi

for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
    command=("${base_command[@]}")
    if [ "$RESUME" = true ] || [ -f "$OUTPUT_DIR/utility_sweep_manifest.json" ]; then
        command+=(--resume)
    fi

    echo "=== Utility sweep attempt $attempt / $MAX_ATTEMPTS ==="
    if (cd "$REPO_ROOT" && "${command[@]}"); then
        if [ "$PREPARE_ONLY" = true ] && [ -f "$OUTPUT_DIR/utility_tasks_prepared.json" ]; then
            echo "=== Counterfactual utility tasks prepared: $OUTPUT_DIR ==="
            exit 0
        elif [ -f "$OUTPUT_DIR/utility_gain_results.json" ] \
            && [ -f "$OUTPUT_DIR/utility_gain_summary.json" ]; then
            echo "=== Counterfactual utility sweep completed: $OUTPUT_DIR ==="
            exit 0
        fi
        echo "Evaluator exited successfully but final result files are missing." >&2
    else
        echo "Utility evaluator exited with an error." >&2
    fi

    if [ "$attempt" -lt "$MAX_ATTEMPTS" ]; then
        echo "Retrying with --resume in $SLEEP_SECONDS seconds ..."
        RESUME=true
        sleep "$SLEEP_SECONDS"
    fi
done

echo "ERROR: utility sweep did not complete after $MAX_ATTEMPTS attempts" >&2
exit 1
