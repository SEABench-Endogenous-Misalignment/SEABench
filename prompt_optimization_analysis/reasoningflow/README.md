# ReasoningFlow for SEABench

This directory is the original ReasoningFlow repository, moved into SEABench and
adapted with a deliberately narrow integration layer. The original staged annotation
algorithm and its original reasoning schemas/prompts remain the primary pipeline:

1. LLM segmentation of each paragraph chunk, followed by the original DP alignment.
2. One batched ReasoningFlow node-role classification.
3. Post-hoc conclusion detection. Here, `conclusion` is minimally generalized to
   include an agent's resolved claim or committed action, not merely a numeric answer.
4. One original edge detection/classification call per generated-response node.
5. The original cleansing passes, except the math/science final-answer extractor.
6. A separate optional safety-label pass over the finalized response nodes.
7. Construction of two DAGs: the original reasoning DAG and a safety-only projection.

The safety DAG does not receive independently annotated edges. It is derived from the
reasoning DAG by retaining safety-labeled nodes, contracting paths whose internal nodes
have no safety label, and applying transitive reduction. Each resulting edge records
its original `relation_path`, `hops`, and `contracted_node_ids`.

Neither annotation pass sees the model's final answer, task/safety pass status, scores,
or evaluator feedback. Those fields are joined from the extracted trace only after all
annotations are finalized, while the two graph artifacts are being serialized.

## Run the SEABench pipeline

Install the requirements into the SEABench environment and set the UVARC key:

```bash
cd SEABench
.venv/bin/pip install -r prompt_optimization_analysis/reasoningflow/requirements.txt
export UVARC_GenAI_API=...
```

To extract the traces and then run the pipeline:

```bash
bash prompt_optimization_analysis/reasoningflow/run_seabench_pipeline.sh \
  --surface short_term_memory \
  --source-root "prompt_optimization/out/kimi_k2_5_short_term_memory/downstream_reruns" \
  --annotator-model "Kimi K2.5" \
  --out-root "prompt_optimization_analysis/out/reasoningflow_kimi_downstream_reruns"
```

To reuse traces already extracted under the output root:

```bash
bash prompt_optimization_analysis/reasoningflow/run_seabench_pipeline.sh \
  --skip-extract \
  --surface short_term_memory \
  --annotator-model "Kimi K2.5" \
  --out-root "prompt_optimization_analysis/out/reasoningflow_kimi_downstream_reruns"
```

The default annotation model is `Kimi K2.5` through the `uvarc` provider profile.
`--surface` is required and accepts `short_term_memory`, `controller_update`, or
`tool-use`; it selects the surface-specific safety-label definitions. Controller-update
labeling also loads the initial `AGENTS.md` instructions from the matching task YAML.
For every surface, the safety judge receives that initial `AGENTS.md` verbatim plus the
active-surface workspace diffs from every upstream run in the replay lineage. Controller
update retains diffs for `AGENTS.md` and `SOUL.md`; short-term-memory retains diffs for
`SHORT_TERM_MEMORY.md` and `SHORT_TERM_MEMORY_COMPACTION.md`; tool use retains diffs for
`TOOLS.md`, `skills/**`, and `created_tools/**`. This separates actual upstream evolution
from seeded placeholder text in the merged workspace. The current downstream run is
excluded.
Useful environment overrides are `MODELS`, `HARMTYPES`, `WORKERS`, `MAX_TOKENS`,
`LIMIT`, `OUT_ROOT`, `TRACES_ROOT`, `MODEL`, `SURFACE`, and `SKIP_EXTRACT=1`.

To force only the safety pass to run again while preserving finalized reasoning
annotations and reasoning graphs, use `--safety-only`. It replaces each existing safety
label file (saving the first prior version under `backup/safety_labels/`) and rebuilds
only the corresponding safety graph:

```bash
bash prompt_optimization_analysis/reasoningflow/run_seabench_pipeline.sh \
  --safety-only \
  --surface short_term_memory \
  --annotator-model "Kimi K2.5" \
  --out-root "prompt_optimization_analysis/out/reasoningflow_kimi_downstream_reruns"
```

Use `--surface tool-use` and the tool-skill output root for that dataset. Traces without
a valid existing reasoning annotation are skipped in this mode rather than triggering
reasoning annotation calls.

Outputs default to `prompt_optimization_analysis/out/reasoningflow_original/`:

- `traces/`: shared extractor output
- `reasoning_annotations/`: finalized upstream-format ReasoningFlow annotations
- `safety_labels/`: the independent optional safety labels
- `reasoning_graphs/`: full original ReasoningFlow DAGs
- `safety_graphs/`: contracted safety-only DAGs
- `backup/edge_retries/`: original annotations saved before targeted edge repair
- `backup/safety_labels/`: legacy/stale safety labels saved before regeneration
- `backup/text_partitions/`: original annotations saved before text-partition repair
- `backup/archives/`: manually created annotation snapshots
- `debug/{reasoning,edge_retries,safety}/`: prompts and raw provider responses

The SEABench wrapper completes each trace before starting the next one: reasoning
node and edge annotation, cleansing, structural validation, safety labeling, and
both graph projections. A restart skips a trace only when all of those artifacts
are present and consistent. With `LIMIT=0`, a final dataset-wide completeness check
runs after the per-trace loop.

The shared extractor is
`prompt_optimization_analysis/cot_extract_traces.py`; both this implementation and
`reasoningflow_adapted/` use it.

If validation reports edge source/destination role violations, retry only the
affected destination nodes with schema feedback:

```bash
python parser/retry_invalid_edges.py \
  --annotations-dir OUTPUT_ROOT/reasoning_annotations \
  --debug-root OUTPUT_ROOT/debug/edge_retries
```

The repair is resumable and writes each original document to
`OUTPUT_ROOT/backup/edge_retries/` before its first successful in-place update.

## Original repository usage

The upstream tools and sample data remain in place below.

## Setup Python Dependencies

```bash
cd web
pip install -r requirements.txt
```

## Start web annotation tool

```bash
./start_server.sh
# localhost:5000 will point to the annotation tool.
# localhost:5001 will point to the annotation guide. You can also check the current main branch's guide at:
# jinulee-v.github.io/reasoningflow
```

## Perform automatic annotation

1. To use VertexAI, add the GCP configuration JSON file in `.env` file. To use other frameworks like Gemini studio, OpenAI, or DeepInfra, (1) open parser/llm_labeler.py and import appropriate package; (2) add the corresponding env key (GOOGLE_API_KEY, OPENAI_API_KEY, DEEPINFRA_API_KEY) to the `.env` file.

```
# .env
GOOGLE_APPLICATION_CREDENTIALS="gcp-config.json"
```


2. Run this code:

```bash
python parser/llm_labeler.py --model gemini-3-flash-preview --raw_data data/v1_raw_data --output_dir OUTPUT_DIR # e.g., data/v1_llm_gemini-3-flash-preview
```

3. After running everything, run this script for cleansing the data:

```bash
./cleanse.sh OUTPUT_DIR # from above script
```
