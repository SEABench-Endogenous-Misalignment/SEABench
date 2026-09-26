# Prompt Optimization

This directory contains the adaptive trajectory-discovery pipeline used to
refine SEABench task sequences. It searches over task-prompt candidates with
paired replays of a self-evolving agent and a non-evolving (`surface=none`)
baseline while preserving the original task intent.

## Task Organization

Benchmark inputs live at:

```text
tasks/<evolution_surface>/<domain>/<harm_type>.yaml
```

Tasks within a sequence are marked as
`optimization.role: upstream` or `optimization.role: downstream`; when that
field is absent, tasks with a `validator.safety_llm` block are treated as
downstream and the others as upstream.

In the paper's terminology, **upstream** tasks are the evolution tasks and
**downstream** tasks are the safety test tasks.

## Refine an Existing Sequence

`refine_existing_sequence.py` is the main entry point. It copies an existing
sequence into an output directory and runs two stages:

1. **Upstream hardening:** refine upstream prompts while requiring useful
   self-evolving behavior, a meaningful persistent update, and first-attempt
   failure by the non-evolving baseline.
2. **Downstream refinement:** replay safety-test tasks from the accepted evolved
   state with further self-evolution frozen, then retry or repair candidates
   that do not satisfy the hard filters.

```bash
export OPENROUTER_API_KEY="<your-key>"

python3 prompt_optimization/refine_existing_sequence.py \
  --task-sequence tasks/controller_update/computer_use/privacy.yaml \
  --output-dir prompt_optimization/out/controller_update_computer_use_privacy \
  --provider-profile openrouter \
  --max-candidates-per-task 5 \
  --downstream-safety-lookahead
```

The principal outputs are:

- `generated_sequence.yaml`: local copy of the input sequence
- `cascade_optimization/`: candidate YAMLs, paired replay artifacts, and logs
- `final_refined_sequence.yaml`: final accepted sequence

## Cascaded Refinement Pipeline

The lower-level cascade optimizes tasks in sequence order. After selecting a
prompt for one task, it replays that accepted task to produce the workspace and
session state used by the next task. Rejected candidate branches are discarded.
At the upstream/downstream boundary, self-evolution can be frozen so downstream
failures are evaluated against the persistent state created upstream rather
than against new downstream updates.

```bash
python3 prompt_optimization/refine_sequence_from_replay_textgrad.py \
  --task-sequence tasks/controller_update/computer_use/privacy.yaml \
  --from-scratch \
  --surface controller_update \
  --provider-profile openrouter \
  --max-candidates-per-task 5 \
  --freeze-self-evolution-after-upstream \
  --output-dir prompt_optimization/out/controller_update_computer_use_privacy_cascade
```

Use `--none-run-dir` and `--self-run-dir` instead of `--from-scratch` to begin
from recorded prior states, `--task-ids` to refine a subset, and `--resume` to
continue an interrupted cascade. The selected sequence is maintained in
`working_sequence.yaml`, with per-task artifacts and state transitions recorded
under the cascade output directory.

The cascade components are intentionally available as separate entry points:

- `refine_task_from_replay_textgrad.py`: refine one task from a recorded prior
  workspace state.
- `refine_sequence_from_replay_textgrad.py`: run the lower-level cascaded replay
  optimizer directly.
- `refine_sequence_from_replay_textgrad_upstream_hardening.py`: run only the
  upstream-hardening stage and downstream refresh/repair logic.
- `refine_sequence_from_replay_textgrad_downstream_retry.py`: reopen and rebuild
  the downstream portion of an existing cascade.

These lower-level tools are primarily useful for debugging, partial reruns, and
targeted analysis.

## Evaluation Semantics

- Candidate evaluations use the real SEABench runtime through `scripts/run.py`.
- Candidate selection compares the original prompt with generated candidates
  and keeps a rewrite only when it satisfies the hard filters and improves the
  objective.
