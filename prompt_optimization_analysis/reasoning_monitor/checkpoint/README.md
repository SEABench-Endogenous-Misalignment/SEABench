# Reference configuration

This directory contains the trained monitor and reported test result:

- `progression_filter/monitor.joblib`: trained progression filter.
- `selector_verifier/stage1.joblib`: trained risk-progression selector.
- `final/selection_and_report.json`: frozen surface-specific decision thresholds and results.
- `split.json`: grouped 216/72/192 train/validation/test assignment.
- `config.json`, `verifier_prompt.md`, and `verifier_schema.json`: fixed method specification.
- `CHECKPOINT.json`: integrity hashes and the reported operating point: 82.3%
  accuracy, 70.9% failure recall, and 9.7% false-positive rate over 192 test
  traces.

The serialized models contain no local paths or experiment names. The top-level
README provides commands for training both models from scratch.
