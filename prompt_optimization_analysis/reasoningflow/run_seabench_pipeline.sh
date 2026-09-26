#!/usr/bin/env bash
# End-to-end original staged ReasoningFlow annotation adapted for SEABench.
set -euo pipefail

CALLER_CWD="$PWD"
RF_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ANALYSIS_ROOT="$(cd "$RF_ROOT/.." && pwd)"
SEABENCH_ROOT="$(cd "$ANALYSIS_ROOT/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$SEABENCH_ROOT/.venv/bin/python}"
OUT_ROOT="${OUT_ROOT:-$ANALYSIS_ROOT/out/reasoningflow_original}"
TRACES_ROOT_FROM_ENV="${TRACES_ROOT+x}"
TRACES_ROOT="${TRACES_ROOT:-$OUT_ROOT/traces}"
MODEL="${MODEL:-Kimi K2.5}"
PROVIDER_PROFILE="${PROVIDER_PROFILE:-uvarc}"
WORKERS="${WORKERS:-8}"
EDGE_RETRY_ATTEMPTS="${EDGE_RETRY_ATTEMPTS:-3}"
MAX_TOKENS="${MAX_TOKENS:-32768}"
LIMIT="${LIMIT:-0}"
SOURCE_ROOT="${SOURCE_ROOT:-}"
SURFACE="${SURFACE:-}"

usage() {
  cat <<'EOF'
Usage: run_seabench_pipeline.sh [options]

Options:
  --out-root PATH           Pipeline output root
  --traces-root PATH        Extracted trace root (default: OUT_ROOT/traces)
  --source-root PATH        Root containing downstream rerun directories
  --harmtypes "NAME ..."    Optional harm-type filter passed to extraction
  --annotator-model NAME    Annotation model name (default: Kimi K2.5)
  --surface NAME            Required: short_term_memory, controller_update, or tool-use
  --provider-profile NAME   Provider profile (default: uvarc)
  --workers N               Concurrent edge workers (default: 8)
  --edge-retry-attempts N   Attempts per invalid edge destination (default: 3)
  --max-tokens N            Maximum output tokens per request (default: 32768)
  --limit N                 Annotate at most N traces; 0 means all (default: 0)
  --safety-only             Redo only safety annotations/graphs from existing reasoning annotations
  --reasoningflow-only      Run only ReasoningFlow node/edge annotation; skip safety annotation
  --skip-extract            Use traces already present under TRACES_ROOT
  -h, --help                Show this help

The UVARC_GenAI_API secret must be supplied through the environment.
Existing environment-variable configuration remains supported; explicit flags win.
EOF
}

traces_root_explicit=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --out-root)
      OUT_ROOT="$2"
      shift 2
      ;;
    --traces-root)
      TRACES_ROOT="$2"
      traces_root_explicit=1
      shift 2
      ;;
    --source-root)
      SOURCE_ROOT="$2"
      shift 2
      ;;
    --harmtypes)
      HARMTYPES="$2"
      shift 2
      ;;
    --annotator-model)
      MODEL="$2"
      shift 2
      ;;
    --surface)
      SURFACE="$2"
      shift 2
      ;;
    --provider-profile)
      PROVIDER_PROFILE="$2"
      shift 2
      ;;
    --workers)
      WORKERS="$2"
      shift 2
      ;;
    --edge-retry-attempts)
      EDGE_RETRY_ATTEMPTS="$2"
      shift 2
      ;;
    --max-tokens)
      MAX_TOKENS="$2"
      shift 2
      ;;
    --limit)
      LIMIT="$2"
      shift 2
      ;;
    --reasoningflow-only)
      REASONINGFLOW_ONLY=1
      shift
      ;;
    --skip-extract)
      SKIP_EXTRACT=1
      shift
      ;;
    --safety-only)
      SAFETY_ONLY=1
      SKIP_EXTRACT=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "$SURFACE" in
  short_term_memory|controller_update|tool-use)
    ;;
  "")
    echo "--surface is required (short_term_memory, controller_update, or tool-use)." >&2
    exit 2
    ;;
  *)
    echo "Unsupported --surface: $SURFACE" >&2
    usage >&2
    exit 2
    ;;
esac

# Unless independently selected, the trace directory follows an overridden
# output root rather than retaining the pre-argument default.
if [[ "$traces_root_explicit" == "0" && -z "$TRACES_ROOT_FROM_ENV" ]]; then
  TRACES_ROOT="$OUT_ROOT/traces"
fi

# Later stages run from RF_ROOT because the original ReasoningFlow scripts use
# repository-relative resources. Resolve user-supplied paths first so changing
# directories cannot change what they refer to.
resolve_path() {
  if [[ "$1" = /* ]]; then
    realpath -m -- "$1"
  else
    realpath -m -- "$CALLER_CWD/$1"
  fi
}

OUT_ROOT="$(resolve_path "$OUT_ROOT")"
TRACES_ROOT="$(resolve_path "$TRACES_ROOT")"
if [[ -n "$SOURCE_ROOT" ]]; then
  SOURCE_ROOT="$(resolve_path "$SOURCE_ROOT")"
fi

if [[ "${SKIP_EXTRACT:-0}" != "1" ]]; then
  extract_args=(--dest "$TRACES_ROOT" --surface "$SURFACE")
  if [[ -n "$SOURCE_ROOT" ]]; then
    extract_args+=(--source-root "$SOURCE_ROOT")
  elif [[ -n "${MODELS:-}" ]]; then
    read -r -a selected_models <<< "$MODELS"
    extract_args+=(--models "${selected_models[@]}")
  fi
  if [[ -n "${HARMTYPES:-}" ]]; then
    read -r -a selected_harmtypes <<< "$HARMTYPES"
    extract_args+=(--harmtypes "${selected_harmtypes[@]}")
  fi
  "$PYTHON_BIN" "$ANALYSIS_ROOT/cot_extract_traces.py" "${extract_args[@]}"
fi

if [[ "${REASONINGFLOW_ONLY:-0}" == "1" && "${SAFETY_ONLY:-0}" == "1" ]]; then
  echo "--reasoningflow-only and --safety-only are mutually exclusive." >&2
  exit 2
fi

if [[ -z "${UVARC_GenAI_API:-}" ]]; then
  echo "UVARC_GenAI_API must be set for ReasoningFlow and safety annotation." >&2
  exit 2
fi

cd "$RF_ROOT"
pipeline_mode_args=()
if [[ "${SAFETY_ONLY:-0}" == "1" ]]; then
  pipeline_mode_args+=(--safety-only)
fi
if [[ "${REASONINGFLOW_ONLY:-0}" == "1" ]]; then
  pipeline_mode_args+=(--reasoningflow-only)
fi
"$PYTHON_BIN" parser/run_seabench_per_trace.py \
  --traces-root "$TRACES_ROOT" \
  --out-root "$OUT_ROOT" \
  --reasoning-debug-root "$OUT_ROOT/debug/reasoning" \
  --edge-debug-root "$OUT_ROOT/debug/edge_retries" \
  --safety-debug-root "$OUT_ROOT/debug/safety" \
  --provider-profile "$PROVIDER_PROFILE" --model "$MODEL" \
  --surface "$SURFACE" \
  --workers "$WORKERS" --edge-retry-attempts "$EDGE_RETRY_ATTEMPTS" \
  --max-tokens "$MAX_TOKENS" --limit "$LIMIT" "${pipeline_mode_args[@]}"
