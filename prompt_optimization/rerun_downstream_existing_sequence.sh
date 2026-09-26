#!/bin/bash
# Rerun downstream in a NEW dir from a completed run's upstream state; source is untouched.
#
#   bash prompt_optimization/rerun_downstream_existing_sequence.sh \
#     --log-dir prompt_optimization/out/<model>/<seq> \
#     --output-dir prompt_optimization/out/<model>/<seq>_downstream_rerun \
#     --task-sequence tasks/<surface>/<category>/<harm-type>.yaml \
#     -- --surface short_term_memory --provider-profile openrouter \
#        --agent-model "moonshotai/kimi-k2.5" --judge-model "moonshotai/kimi-k2.5" \
#        --max-candidates-per-task 5 --downstream-safety-lookahead --downstream-standalone
#
#   # continue an interrupted rerun (no re-stage, --log-dir not needed):
#   bash prompt_optimization/rerun_downstream_existing_sequence.sh --resume \
#     --output-dir prompt_optimization/out/<model>/<seq>_downstream_rerun \
#     --task-sequence tasks/<surface>/<category>/<harm-type>.yaml \
#     -- --surface short_term_memory --provider-profile openrouter \
#        --agent-model "moonshotai/kimi-k2.5" --judge-model "moonshotai/kimi-k2.5" \
#        --max-candidates-per-task 5 --downstream-safety-lookahead --downstream-standalone

set -u
shopt -s nullglob

LOG_DIR=""
OUTPUT_DIR=""
TASK_SEQUENCE=""
RESUME=false
FWD=()
while [ $# -gt 0 ]; do
    case "$1" in
        --log-dir)        LOG_DIR="$2"; shift 2 ;;
        --output-dir)     OUTPUT_DIR="$2"; shift 2 ;;
        --task-sequence)  TASK_SEQUENCE="$2"; shift 2 ;;
        --resume)         RESUME=true; shift ;;
        --)               shift; FWD=("$@"); break ;;
        *)                echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

[ -n "$OUTPUT_DIR" ] && [ -n "$TASK_SEQUENCE" ] || {
    echo "Need --output-dir and --task-sequence" >&2; exit 1; }
if [ "$RESUME" = false ] && [ -z "$LOG_DIR" ]; then
    echo "Need --log-dir (or pass --resume to continue an already-staged --output-dir)" >&2
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DST_CASCADE="$OUTPUT_DIR/cascade_optimization"
DST_LOG="$DST_CASCADE/sequence_optimization_log.jsonl"

id_of() { grep -oE '"task_id":[[:space:]]*"[^"]+"' | sed -E 's/.*"([^"]+)"$/\1/'; }

if [ "$RESUME" = true ]; then
    [ -f "$DST_CASCADE/working_sequence.yaml" ] || {
        echo "Cannot resume: missing $DST_CASCADE/working_sequence.yaml (nothing staged in --output-dir)" >&2
        exit 1
    }
    echo "Resume mode: continuing downstream rerun in $OUTPUT_DIR ..."
    exec bash "$REPO_ROOT/scripts/run_refinement_existing_sequence.sh" \
        --task-sequence "$TASK_SEQUENCE" --output-dir "$OUTPUT_DIR" "${FWD[@]}"
fi

SRC_CASCADE="$LOG_DIR/cascade_optimization"
SRC_LOG="$SRC_CASCADE/sequence_optimization_log.jsonl"
[ -f "$SRC_LOG" ] || { echo "Missing $SRC_LOG" >&2; exit 1; }

if [ -e "$OUTPUT_DIR" ] && [ "$LOG_DIR" -ef "$OUTPUT_DIR" ]; then
    echo "--log-dir and --output-dir must be different directories" >&2
    exit 1
fi
if [ -e "$DST_LOG" ]; then
    echo "$DST_LOG already exists; pass --resume to continue it, or choose a fresh --output-dir" >&2
    exit 1
fi

mapfile -t UP_IDS < <(grep -E '"role":[[:space:]]*"upstream"' "$SRC_LOG" | id_of | sort -u)
LAST_UP="$(grep -E '"role":[[:space:]]*"upstream"' "$SRC_LOG" | tail -1 | id_of)"
[ -n "$LAST_UP" ] || { echo "no upstream rows in $SRC_LOG" >&2; exit 1; }

if grep -qE '"role":[[:space:]]*"downstream"' "$SRC_LOG"; then
    echo "Note: source has downstream rows; they are left untouched in $LOG_DIR and dropped from the copy."
else
    echo "Note: source has no downstream rows; staging from upstream state alone."
fi

SRC_WS=("$SRC_CASCADE"/*"$LAST_UP"/refined_sequence.yaml)
if [ "${#SRC_WS[@]}" -eq 0 ] && [ -f "$SRC_CASCADE/working_sequence.yaml" ]; then
    SRC_WS=("$SRC_CASCADE/working_sequence.yaml")
fi
[ "${#SRC_WS[@]}" -gt 0 ] || {
    echo "no $SRC_CASCADE/*$LAST_UP/refined_sequence.yaml and no $SRC_CASCADE/working_sequence.yaml" >&2
    exit 1
}

mkdir -p "$DST_CASCADE"

grep -E '"role":[[:space:]]*"upstream"' "$SRC_LOG" > "$DST_LOG"
cp -p "${SRC_WS[0]}" "$DST_CASCADE/working_sequence.yaml"
for tid in "${UP_IDS[@]}"; do
    for d in "$SRC_CASCADE"/*"$tid"; do
        [ -d "$d" ] && cp -a "$d" "$DST_CASCADE/"
    done
done
[ -f "$LOG_DIR/generated_sequence.yaml" ] && cp -p "$LOG_DIR/generated_sequence.yaml" "$OUTPUT_DIR/generated_sequence.yaml"

echo "Staged $OUTPUT_DIR from $LOG_DIR (post-$LAST_UP state); launching downstream rerun..."
exec bash "$REPO_ROOT/scripts/run_refinement_existing_sequence.sh" \
    --task-sequence "$TASK_SEQUENCE" --output-dir "$OUTPUT_DIR" "${FWD[@]}"
