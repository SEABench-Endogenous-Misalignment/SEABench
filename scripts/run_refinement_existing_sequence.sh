#!/bin/bash

MAX_ATTEMPTS=20
SLEEP_SECONDS=5

# Parse --output-dir and check for --resume from the forwarded args
OUTPUT_DIR=""
HAS_RESUME=false
args=("$@")
for i in "${!args[@]}"; do
    if [ "${args[$i]}" = "--output-dir" ]; then
        OUTPUT_DIR="${args[$((i+1))]}"
    fi
    if [ "${args[$i]}" = "--resume" ]; then
        HAS_RESUME=true
    fi
done

if [ -z "$OUTPUT_DIR" ]; then
    echo "Error: --output-dir not found in arguments."
    exit 1
fi

FINAL_SEQUENCE="$OUTPUT_DIR/final_refined_sequence.yaml"

for attempt in $(seq 1 $MAX_ATTEMPTS); do
    echo "=== Attempt $attempt / $MAX_ATTEMPTS ==="

    if [ -d "$OUTPUT_DIR" ] && [ "$HAS_RESUME" = false ]; then
        echo "Running: python3 prompt_optimization/refine_existing_sequence.py $* --resume"
        python3 prompt_optimization/refine_existing_sequence.py "$@" --resume
    else
        echo "Running: python3 prompt_optimization/refine_existing_sequence.py $*"
        python3 prompt_optimization/refine_existing_sequence.py "$@"
    fi
    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ] && [ -f "$FINAL_SEQUENCE" ]; then
        echo "=== Success: run completed cleanly and final_refined_sequence.yaml exists. ==="
        exit 0
    fi

    if [ $EXIT_CODE -eq 0 ] && [ ! -f "$FINAL_SEQUENCE" ]; then
        echo "Exit code 0 but final_refined_sequence.yaml not found — treating as incomplete."
    else
        echo "Command exited with code $EXIT_CODE."
    fi

    if [ $attempt -lt $MAX_ATTEMPTS ]; then
        echo "Retrying in $SLEEP_SECONDS seconds..."
        sleep $SLEEP_SECONDS
    fi
done

echo "=== Failed: max attempts ($MAX_ATTEMPTS) reached without successful completion. ==="
exit 1
