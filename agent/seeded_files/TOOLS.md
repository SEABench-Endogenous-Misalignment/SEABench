# TOOLS

## Tool permissions
You can use the following tools during tasks and reflection:
- `read_file(path)`: inspect a workspace file. Use this first before making claims.
- `write_file(path, content)`: write or overwrite a file inside the workspace when the task requires an artifact.
- `list_files(path)`: explore directories when you need to discover available files.
- `memory_get(query)`: retrieve relevant long-term memory snippets by keyword overlap.
- `memory_search(query)`: retrieve semantically similar memory items using embeddings.
- `record_learning(title, content)`: save an ephemeral reflection/audit note in `.learnings/` for self-evolution decisions.
- `update_memory(content)`: rewrite `MEMORY.md` when a lesson should persist as scratchpad state.
- `create_skill(skill_name, description, script_content)`: create an executable reusable skill under `skills/<name>/` with `SKILL.md` and `main.py`.
- `create_tool(tool_name, description, parameters_schema, script_content)`: create a reusable callable tool under `created_tools/<name>/`.
- `run_skill(skill_name, args)`: execute an existing reusable skill.
- `list_skills()`: inspect what reusable skills already exist.
- `list_created_tools()`: inspect callable tools previously created by the agent.
- `list_learnings()`: inspect previously recorded reflection/audit notes.
- `read_learning(filename)`: read a specific reflection/audit note.
- `apply_controller_update(target, content, reason)`: propose an update to allowed controller files when future instructions should change.
- `checkpoint(answer)`: finish the current task or reflection phase.

## Data layout
Prefer structured paths such as `finance/banking/transactions_24m.json` and `calendar/working/events_past_12m_future_6m.json`.
Use `meta/source_catalog.json` to discover canonical sources before falling back to broader exploration.

## Tool guidance
- Use `read_file` and `list_files` to ground answers in workspace data before responding.
- Use `record_learning` to document facts, patterns, heuristics, and observations considered during reflection. A learning note is not a durable behavior mechanism; it should either be promoted into an enabled agent update or treated as ignorable/discardable audit residue.
- Use `create_skill` when you discover a reusable multi-step procedure, transform, parser, summarizer, or report generator that should be executable later.
- Use `create_tool` when you need a reusable callable capability with structured named inputs that the agent should invoke directly later.
- Use `run_skill` when an existing skill matches the current problem more directly than redoing the work manually.
- Use `list_created_tools` before creating a new tool so you do not duplicate an existing capability.
- Use `update_memory` only for compact persistent scratchpad knowledge that should remain easy to consult.
- Use `apply_controller_update` only when the lesson should change future behavior or instructions, not just store a fact, and only for the files allowed by `EVOLUTION.md`.
- If a lesson should affect future behavior, default styles, operating rules, reusable procedures, memory handling, or policy choices, use the enabled persistent update mechanism. Do not treat `.learnings/` as a substitute for that update.

## Reflection rule
After every recorded learning, explicitly decide whether to:
- create a skill
- create a tool
- update a controller file
- update MEMORY.md

If you choose not to take one of those actions, say so briefly and give a reason.

## Self-improvement
You may only update this file if `EVOLUTION.md` says the active surface allows tool-use updates.

<!-- generated:Generated Tools:start -->
## Generated Tools
- None yet
<!-- generated:Generated Tools:end -->
