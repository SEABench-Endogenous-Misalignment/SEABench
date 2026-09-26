# Reference configuration

This directory contains the fixed configuration and reported validation result:

- `progression_filter/monitor.joblib`: trained progression filter.
- `selector_verifier/stage1.joblib`: trained risk-progression selector.
- `final/selection_and_report.json`: frozen decision thresholds and validation results.
- `split.json`: grouped 216/72/192 train/validation/test assignment.
- `config.json`, `verifier_prompt.md`, and `verifier_schema.json`: fixed method specification.
- `CHECKPOINT.json`: integrity hashes and the reported operating point.

The serialized models exclude local paths, experiment names, discarded training
vocabulary, and model-search history. The top-level README also provides commands
for training both models from scratch.
