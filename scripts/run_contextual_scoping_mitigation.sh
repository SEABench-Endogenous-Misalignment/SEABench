#!/bin/bash

set -u

MAX_ATTEMPTS=20
SLEEP_SECONDS=5
OUTPUT_DIR=""
HAS_RESUME=false
ARGS=("$@")

for i in "${!ARGS[@]}"; do
    if [ "${ARGS[$i]}" = "--output-dir" ]; then
        OUTPUT_DIR="${ARGS[$((i + 1))]:-}"
    fi
    if [ "${ARGS[$i]}" = "--resume" ]; then
        HAS_RESUME=true
    fi
done

if [ -z "$OUTPUT_DIR" ]; then
    echo "Error: pass the normal refinement arguments, including --output-dir PATH." >&2
    exit 1
fi

if [ "$HAS_RESUME" = true ]; then
    echo "Error: do not pass --resume; this wrapper starts from scratch." >&2
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "$OUTPUT_DIR" = /* ]]; then
    OUTPUT_PATH="$OUTPUT_DIR"
else
    OUTPUT_PATH="$REPO_ROOT/$OUTPUT_DIR"
fi

if [ -e "$OUTPUT_PATH" ]; then
    echo "Refusing to resume an existing run: $OUTPUT_DIR" >&2
    echo "Choose a new output directory for a from-scratch mitigation run." >&2
    exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
FINAL_SEQUENCE="$OUTPUT_PATH/final_refined_sequence.yaml"

export CONTEXTUAL_SCOPING_DEFENSE=1

BASE_CMD=(
    "$PYTHON_BIN" -u "$REPO_ROOT/prompt_optimization/refine_existing_sequence.py"
    "${ARGS[@]}"
)

for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
    echo "=== Attempt $attempt / $MAX_ATTEMPTS ==="

    if [ "$attempt" -eq 1 ]; then
        "${BASE_CMD[@]}"
    else
        "${BASE_CMD[@]}" --resume
    fi
    EXIT_CODE=$?

    if [ "$EXIT_CODE" -eq 0 ] && [ -f "$FINAL_SEQUENCE" ]; then
        echo "=== Success: from-scratch mitigation run completed. ==="
        exit 0
    fi

    if [ "$attempt" -lt "$MAX_ATTEMPTS" ]; then
        echo "Attempt failed or incomplete; resuming in $SLEEP_SECONDS seconds..."
        sleep "$SLEEP_SECONDS"
    fi
done

echo "=== Failed after $MAX_ATTEMPTS attempts. ===" >&2
exit 1
