# Agent Tools

All tools are registered in `scripts/run.py` and passed to the model via the OpenAI `tools` field on every API call.

---

## Filesystem (`agent/tools/core/filesystem.py`)

General-purpose file access within the sandbox. All paths are resolved through `SandboxGuard`, which prevents escaping the workspace root.

| Tool | Description |
|---|---|
| `read_file(path)` | Read the full contents of a file. |
| `write_file(path, content)` | Write content to a file, creating parent directories as needed. Use this for task data files (arbitrary workspace paths not covered by the specialized write tools). |
| `list_files(directory=".")` | List files and subdirectories in a workspace directory. |

---

## Memory Retrieval (`agent/tools/core/memory.py`)

Delegates to extension scripts seeded into the workspace at `extensions/memory_get/` and `extensions/memory_search/`. The scripts are agent-editable, so the agent can improve its own retrieval logic.

| Tool | Description |
|---|---|
| `memory_get(query, limit=3)` | Keyword-based search over `memory/`. Returns up to `limit` file snippets ranked by term overlap. |
| `memory_search(query, limit=3)` | Semantic search over `memory/`. Returns up to `limit` file snippets ranked by embedding similarity. Preferred for conceptual queries. |

---

## Compaction (`agent/tools/core/compaction.py`)

Automatic background extension (but can also be a callable tool). Compaction runs automatically when the conversation context approaches the model's token limit.

The compaction logic lives in `extensions/compaction/compaction.py` inside the workspace — the agent can read and rewrite this file to evolve its own compaction strategy.

| Tool | Description |
|---|---|
| `compact()` | Manually trigger compaction immediately, regardless of the current token count. Summarises oldest messages into the rolling context summary and flushes them to memory. |



## Self-Improvement (`agent/tools/core/self_improve.py`)

Tools for the agent to persist knowledge, create reusable procedures, and update its own controller files. Each targets a specific path in the workspace.

### Knowledge persistence

| Tool | Target | Description |
|---|---|---|
| `record_learning(title, content)` | `.learnings/<date>-<title>.md` | Save an ephemeral reflection/audit note for self-evolution decisions. Required at least once per task reflection phase; not a durable behavior mechanism. Recorded lessons should be promoted into enabled agent updates or treated as ignorable/discardable audit residue. |
| `update_memory(content)` | `MEMORY.md` | Rewrite the persistent memory scratchpad (survives across tasks). |

### Skills

| Tool | Target | Description |
|---|---|---|
| `create_skill(skill_name, description, script_content)` | `skills/<name>/main.py` | Write a reusable Python script. The script receives `WORKSPACE_DIR` as an env var and any args via `sys.argv`. |
| `run_skill(skill_name, args=[])` | — | Execute a previously created skill and return its stdout. |
| `list_skills()` | — | List all skills with their descriptions. |

### Learnings

| Tool | Description |
|---|---|
| `list_learnings()` | List all saved reflection/audit notes in `.learnings/`. |
| `read_learning(filename)` | Read a specific reflection/audit note. |

### Controller updates

| Tool | Target | Description |
|---|---|---|
| `apply_controller_update(target, content, reason)` | `SOUL.md`, `TOOLS.md`, `SKILLS.md`, or `MEMORY.md` | Propose a rewrite of a controller file. Runs through a safety gate that rejects forbidden markers, suspicious phrases, large diffs, and token expansion before writing. |

---

## Control (`agent/tools/registry.py`)

| Tool | Description |
|---|---|
| `checkpoint(answer)` | Submit the final answer for the current task. Ends the task immediately. Must be called exactly once per task (and once per reflection phase). |
