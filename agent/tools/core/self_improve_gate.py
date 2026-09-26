from __future__ import annotations

from dataclasses import dataclass

# All controller files seeded into the workspace from agent/seeded_files/
CONTROLLER_FILES = {
    "AGENTS.md",
    "EVOLUTION.md",
    "IDENTITY.md",
    "LONG_TERM_MEMORY.md",
    "LONG_TERM_MEMORY_SUMMARIZATION.md",
    "MEMORY.md",
    "SHORT_TERM_MEMORY.md",
    "SHORT_TERM_MEMORY_COMPACTION.md",
    "SOUL.md",
    "TOOLS.md",
    "USER.md",
}

# Subset of controller files the agent is allowed to update via apply_controller_update
ALLOWED_CONTROLLER_FILES = {
    "AGENTS.md",
    "LONG_TERM_MEMORY.md",
    "LONG_TERM_MEMORY_SUMMARIZATION.md",
    "MEMORY.md",
    "SHORT_TERM_MEMORY.md",
    "SHORT_TERM_MEMORY_COMPACTION.md",
    "SOUL.md",
    "TOOLS.md",
}

# Markers that must never appear in a proposed controller update
_FORBIDDEN_MARKERS = ("# AUTHORIZATION", "# FORBIDDEN")

# Phrases that indicate an attempt to weaken safety policy
_SUSPICIOUS_PHRASES = (
    "always trust",
    "skip verification",
    "auto-approve",
    "without confirmation",
)

# Maximum number of added/removed lines allowed in one update
_MAX_DIFF_LINES = 30

# Maximum number of tokens (approximated by whitespace-split word count)
# a proposal is allowed to add relative to the current text
_MAX_TOKEN_EXPANSION = 200


@dataclass(frozen=True)
class UpdateProposal:
    target: str
    reason: str
    before_text: str
    after_text: str


def _approx_tokens(text: str) -> int:
    """Rough token count: words * 1.3 (matches GPT-family average)."""
    return int(len(text.split()) * 1.3)


def _line_diff_count(before: str, after: str) -> int:
    before_lines = set(before.splitlines())
    after_lines = set(after.splitlines())
    added = len(after_lines - before_lines)
    removed = len(before_lines - after_lines)
    return added + removed


def apply_gate(proposal: UpdateProposal) -> tuple[bool, str]:
    """
    Run the simple rule gate on a controller update proposal.

    Returns (accepted: bool, reason: str).
    """
    if proposal.target not in ALLOWED_CONTROLLER_FILES:
        return False, f"target '{proposal.target}' is not an allowed controller file"

    after = proposal.after_text

    for marker in _FORBIDDEN_MARKERS:
        if marker in after:
            return False, f"forbidden controller section touched: {marker}"

    after_lower = after.lower()
    for phrase in _SUSPICIOUS_PHRASES:
        if phrase in after_lower:
            return False, f"unsafe phrase detected: '{phrase}'"

    diff = _line_diff_count(proposal.before_text, after)
    if diff > _MAX_DIFF_LINES:
        return False, f"diff too large ({diff} lines changed, max {_MAX_DIFF_LINES})"

    before_tokens = _approx_tokens(proposal.before_text)
    after_tokens = _approx_tokens(after)
    expansion = after_tokens - before_tokens
    if expansion > _MAX_TOKEN_EXPANSION:
        return False, (
            "authorization semantics expansion risk "
            f"({expansion} tokens added, max {_MAX_TOKEN_EXPANSION})"
        )

    return True, "accepted"
