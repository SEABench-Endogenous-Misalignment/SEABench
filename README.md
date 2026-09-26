# SEABench

SEABench is a benchmark for studying endogenous misalignment in self-evolving
LLM agents. It contains 48 longitudinal task sequences spanning multiple
self-evolution surfaces, task domains, and harm types in a personal-assistant
environment. The benchmark pairs self-evolving runs with non-evolving controls
and uses an adaptive trajectory-discovery pipeline to find safety failures while
preserving the intent of the original tasks.

The primary paper experiments refine the existing sequences under `tasks/` with
the prompt-optimization pipeline described below.

## Repo Layout

- `agent/`: agent implementation, controller surface, and tools.
- `analysis_tools/`: benchmark analysis and tracking helpers.
- `prompt_optimization/`: adaptive task-sequence refinement, replay, and utility-evaluation pipelines.
- `prompt_optimization_analysis/`: Chain-of-Thought analysis and mitigation scripts
- `runtime/`: shared runtime configuration and task loading.
- `sandbox/`: workspace seeding utilities and Apptainer definition.
- `scripts/`: benchmark, smoke-test, and reset entrypoints.
- `configs/`: provider profiles.
- `env_assets/`: immutable environment seed data and generation utilities, main environment used for experiments.
- `env_assets_utility_test/`: counterfactual environments with alternate users for testing utility
- `tasks/`: benchmark task manifests split by evolution surface
- `artifacts/`: ignored run outputs, created on demand.

Task sequence YAML can also include per-sequence and per-task setup blocks to override controller files and tool exposure. See `docs/task_setup.md`.

## Environment Setup

Install the dependencies and configure the provider key:

```bash
./scripts/setup_venv.sh
source .venv/bin/activate
export OPENROUTER_API_KEY="<your-key>"
```

## Main Experiment: Refine an Existing Sequence

The main experiment starts from one of the task-sequence YAML files in `tasks/`
and searches over prompt candidates using paired replays of a self-evolving
agent and a non-evolving (`surface=none`) baseline. The central entry point is
`prompt_optimization/refine_existing_sequence.py`.


```bash
python3 prompt_optimization/refine_existing_sequence.py \
  --task-sequence tasks/controller_update/computer_use/privacy.yaml \
  --output-dir prompt_optimization/out/controller_update_computer_use_privacy \
  --surface controller_update \
  --max-candidates-per-task 5 \
  --downstream-safety-lookahead \
  --provider-profile openrouter \
  --agent-model openai/gpt-5.6-luna \
  --judge-model openai/gpt-5.6-luna
```

To resume an interrupted run, add the `--resume` flag to the same command. A run is complete
when `final_refined_sequence.yaml` has been produced.


For long-running experiments, invoke it through the retry shell wrapper
`scripts/run_refinement_existing_sequence.sh`.

```bash
./scripts/run_refinement_existing_sequence.sh \
  --task-sequence tasks/controller_update/computer_use/privacy.yaml \
  --output-dir prompt_optimization/out/controller_update_computer_use_privacy \
  --surface controller_update \
  --max-candidates-per-task 5 \
  --downstream-safety-lookahead \
  --provider-profile openrouter \
  --agent-model openai/gpt-5.6-luna \
  --judge-model openai/gpt-5.6-luna
```


The pipeline infers the evolution surface and sequence ID from the input YAML
unless `--surface` or `--sequence-id` is supplied. It then runs two stages:

In this repository, **upstream** refers to the evolution tasks in the paper,
while **downstream** refers to the safety test tasks in the paper.

1. **Upstream hardening.** Upstream tasks are refined while retaining the normal
   contrastive checks and requiring the non-evolving baseline to fail on its
   first attempt. Accepted trajectories must still exhibit useful task
   completion and a meaningful persistent self-evolution update.
2. **Downstream refinement.** Downstream safety-test tasks are replayed from the
   state created by the accepted upstream trajectory. Further self-evolution is
   frozen so observed downstream failures can be attributed to persistent
   upstream changes; tasks that do not meet the hard filters are retried and
   repaired.

Candidate search defaults to the TextGrad backend. The main search budget is
set with `--max-candidates-per-task`. Use `--optimizer-model` to override the textgrad rewriter from their provider profiles.

Each output directory contains a copy of the input as
`generated_sequence.yaml`, staged artifacts and logs under
`cascade_optimization/`, and the final accepted sequence as
`final_refined_sequence.yaml`. The source YAML is left unchanged unless
`--in-place` is explicitly passed. See `prompt_optimization/README.md` for the
lower-level refinement entry points, acceptance rules, and advanced options.

## Inital Task Sequence Creation
For creation of new task sequences, see `docs/task_sequence_creation.md`.

## Runtime Boundary

- Immutable seed data: `env_assets/`
- Immutable controller seed: `agent/seeded_files/`
- Mutable runtime workspace: `artifacts/run_<timestamp>/workspace/`
- Mutable run artifacts: `artifacts/run_<timestamp>/`

Before each run, a fresh per-run workspace is reset from immutable seeds.
The live workspace is seeded from the structured directory tree under `env_assets/`;
Runtime state is created per run under `artifacts/run_<timestamp>/`, including that run's `workspace/`, logs, checkpoints, and analysis outputs.

## General Task Sequence Runner

The following workflow runs task sequences without the adaptive refinement loop.
It remains useful for direct benchmark execution, smoke tests, and evaluating
already-refined sequences.

```bash
# Create a local virtual environment
./scripts/setup_venv.sh
source .venv/bin/activate

# Reset runtime state
./scripts/reset_runtime.sh

# Smoke test (builds/uses sandbox image and validates runtime contract)
./scripts/test_smoke.sh

# Run benchmark loop
export OPENROUTER_API_KEY="<your-key>"
./scripts/run.sh

# Run only one configured task sequence
./scripts/run.sh --run-config configs/runs/controller_update_computer_use_privacy.yaml
```

## Chain-of-Thought Analysis and Mitigation
Refer `prompt_optimization_analysis/reasoningflow/README` for details on running the post-hoc chain of thought analysis, and `prompt_optimization_analysis/reasoning_monitor/README` for the mitigation experiments.
