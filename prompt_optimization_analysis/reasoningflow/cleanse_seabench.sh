#!/usr/bin/env bash
# Original ReasoningFlow cleansing sequence, excluding science-answer extraction.
set -euo pipefail

RF_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${1:-$RF_ROOT/../out/reasoningflow_original/reasoning_annotations}"
PYTHON_BIN="${PYTHON_BIN:-$RF_ROOT/../../.venv/bin/python}"

cd "$RF_ROOT"
"$PYTHON_BIN" parser/cleanse_scripts/cleanse_weird_segmentation.py --data-dir "$OUTPUT_DIR"
"$PYTHON_BIN" parser/cleanse_scripts/cleanse_nonexisting_labels.py --data-dir "$OUTPUT_DIR" scan
"$PYTHON_BIN" parser/cleanse_scripts/cleanse_think_metatags.py --data-dir "$OUTPUT_DIR"
"$PYTHON_BIN" parser/cleanse_scripts/cleanse_common_errors.py --pattern 1 --data-dir "$OUTPUT_DIR"
"$PYTHON_BIN" parser/cleanse_scripts/cleanse_common_errors.py --pattern 2 --data-dir "$OUTPUT_DIR"
"$PYTHON_BIN" parser/cleanse_scripts/cleanse_reorder_nodes.py --data-dir "$OUTPUT_DIR"

# update_final_answer.py is deliberately omitted: it is tied to the original
# math/science datasets and would expose answer material that annotation never sees.
