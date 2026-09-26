# Reasoning-Trace Safety Monitor

## Install

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .

export OPENAI_API_KEY="..."
export REASONING_MONITOR_MODEL="YOUR_VERIFIER_MODEL"
```

## Input data

```text
DATA/
  controller_update/
    reasoning_graphs/**/*.json
    risk_thread_safety_labels/**/*.json
  short_term_memory/
    reasoning_graphs/**/*.json
    risk_thread_safety_labels/**/*.json
  tools_skills/
    reasoning_graphs/**/*.json
    risk_thread_safety_labels/**/*.json
```

Each annotation file must have the same relative path as its reasoning graph.

## Train and validate

```bash
reasoning-monitor train \
  --data-root DATA \
  --output-dir RUN
```

This creates the grouped split, trains the monitor, selects its thresholds on
the validation set, and freezes it for testing.

## Test

```bash
reasoning-monitor test \
  --data-root DATA \
  --output-dir RUN
```

Results are written to:

```text
RUN/final/test_report.json
RUN/final/test_predictions.jsonl
```

## Train and test with one command

```bash
reasoning-monitor run \
  --data-root DATA \
  --output-dir RUN
```

## Generate a verifier prompt

```bash
derive-verifier-prompt \
  --model YOUR_PROMPT_WRITER_MODEL \
  --output-prompt verifier_prompt.md
```

Use it with `--verifier-prompt verifier_prompt.md` when running the monitor.
