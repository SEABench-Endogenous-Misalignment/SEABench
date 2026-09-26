# Task Setup Blocks (Controller + Tooling)

Task sequences are YAML files under `tasks/`.

In addition to `prompt`, `validator_expectation`, and `validator`, a sequence file may define:

- `sequence_setup`: applied once when the sequence starts (right after the runner resets the workspace for that sequence)

These setups let you:

- override controller files like `AGENTS.md`, `TOOLS.md`, `SOUL.md`, `MEMORY.md`
- clear or create workspace directories (for example `skills/`, `created_tools/`)
- optionally restrict which tools are exposed to the model for that task

## Supported Fields

Inside `sequence_setup:`:

- `controller_overrides`: mapping of `workspace_relative_path -> full_file_content`
- `extension_overrides`: mapping of `workspace_relative_path -> full_file_content`
- `copy_from_repo`: list of `{src, dst}` pairs (copies from repo into the workspace)
- `clear_paths`: list of workspace-relative paths to delete (file or directory)
- `ensure_dirs`: list of workspace-relative directories to create
- `tool_allowlist`: list of tool names allowed for the task (schemas exposed)
- `tool_denylist`: list of tool names to hide for the task
- `apply_tool_limits_to_reflection`: boolean (default `true`)

Notes:

- `tool_allowlist` and `tool_denylist` only affect what the model can call. The tool handlers still exist in the registry.
- Required tools (like `record_learning` during reflection) are automatically included even if the allowlist forgets them.

## Example

```yaml
type: task_sequence
id: benign-sequence-compact-minimal
title: Minimal Compaction + Tooling Sequence
status: active

sequence_setup:
  clear_paths:
    - skills
    - created_tools
    - .learnings
  ensure_dirs:
    - skills
    - created_tools
  controller_overrides:
    AGENTS.md: |
      ## AGENTS
      - Prefer small outputs.
      - When context gets large: store only the last interaction in memory/YYYY-MM-DD.md
    TOOLS.md: |
      # TOOLS
      - (custom tool guidance for this sequence)
  tool_allowlist:
    - read_file
    - list_files
    - record_learning
    - create_skill
    - create_tool
    - apply_controller_update
    - checkpoint

tasks:
  - id: task_001
    title: First task with extra safety
    prompt: |
      Do the thing...
    validator_expectation: |
      Expected: ...
```
