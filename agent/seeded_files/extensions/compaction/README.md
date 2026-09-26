# Compaction

**Type**: Automatic background extension (but can also be a callable tool)

Compaction runs automatically when the conversation context approaches the model's token limit. It summarises the oldest messages into a rolling structured summary that is re-injected into the system prompt, then discards those messages from the active context.

## What it does

1. **Detects overflow** — when estimated request tokens exceed `compaction_trigger_ratio × context_window_tokens`, compaction fires.
2. **Splits history** — keeps the most recent messages intact; selects older messages for summarisation.
3. **Summarises** — calls the model with a structured summarisation prompt that updates a running `## Goal / ## Progress / ## Key Decisions / ## Next Steps / ## Critical Context` document.
4. **Flushes to memory** — writes the compacted summary into `memory/YYYY-MM-DD.md` between `<!-- compacted-context:start -->` and `<!-- compacted-context:end -->` markers so it persists across sessions.
5. **Rebuilds the system prompt** — the compacted summary is prepended to the next system prompt so situational awareness is preserved.

## Effect on you

- Compaction is transparent — you will not observe it happening mid-turn.
- Important progress, decisions, and context are preserved in the structured summary.
- After each compaction round, `memory/YYYY-MM-DD.md` is updated. You can inspect it with `read_file`.
