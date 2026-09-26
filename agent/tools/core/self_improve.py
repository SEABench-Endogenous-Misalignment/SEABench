from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agent.tools.registry import ToolRegistry
from agent.tools.core.self_improve_gate import (
    UpdateProposal,
    apply_gate,
)
from analysis_tools.self_improvement.tracker import EvolutionTracker
from runtime.evolution import (
    allowed_controller_targets,
    allowed_self_improvement_actions,
    validate_evolution_surface,
)
from sandbox.workspace import SandboxGuard
from agent.tools.base import ToolResult


def _safe_name(value: str) -> str:
    """Sanitise a string for use as a filename or directory name."""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-._") or "unnamed"


def make_self_improve_tools(
    workspace_dir: Path,
    tracker: EvolutionTracker,
    registry: ToolRegistry | None = None,
    evolution_surface: str = "controller_update",
) -> list[tuple[str, Any, str, dict]]:
    """
    Returns (name, handler, description, parameters) tuples for all self-improvement tools.

    Skills are Python scripts stored at workspace/skills/<name>/main.py.
    Each skill also has workspace/skills/<name>/SKILL.md describing what it does.
    When run, they receive:
      - WORKSPACE_DIR env var pointing to the workspace root
      - Any args you pass as sys.argv[1:]
      - Expected to print results to stdout
    """

    guard = SandboxGuard(workspace_dir)
    surface = validate_evolution_surface(evolution_surface)
    allowed_actions = set(allowed_self_improvement_actions(surface))
    writable_targets = tuple(allowed_controller_targets(surface))
    if not allowed_actions:
        return []

    def _update_generated_section(path_str: str, header: str, items: list[str]) -> None:
        path = guard.resolve(path_str)
        marker_start = f"<!-- generated:{header}:start -->"
        marker_end = f"<!-- generated:{header}:end -->"
        body = "\n".join(items) if items else "- None yet"
        section = f"{marker_start}\n## {header}\n{body}\n{marker_end}"
        current = path.read_text(encoding="utf-8") if path.exists() else ""
        if marker_start in current and marker_end in current:
            start_idx = current.index(marker_start)
            end_idx = current.index(marker_end) + len(marker_end)
            updated = current[:start_idx].rstrip() + "\n\n" + section + "\n"
        else:
            updated = current.rstrip() + ("\n\n" if current.strip() else "") + section + "\n"
        path.write_text(updated, encoding="utf-8")

    def _refresh_tool_registry_doc() -> None:
        created_dir = guard.resolve("created_tools")
        items: list[str] = []
        for tool_dir in sorted(path for path in created_dir.iterdir() if path.is_dir()):
            manifest_path = tool_dir / "manifest.json"
            if not manifest_path.exists():
                continue
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            name = str(manifest.get("name", tool_dir.name))
            description = str(manifest.get("description", "")).strip()
            suffix = f": {description}" if description else ""
            items.append(f"- `{name}`{suffix}")
        _update_generated_section("TOOLS.md", "Generated Tools", items)

    # ------------------------------------------------------------------ #
    # record_learning                                                       #
    # ------------------------------------------------------------------ #
    def record_learning(title: str, content: str) -> ToolResult:
        try:
            stem = _safe_name(title)[:50]
            date = datetime.now(UTC).date().isoformat()
            filename = f"{date}-{stem}.md"
            path = guard.resolve(f".learnings/{filename}")
            path.write_text(
                f"# {title}\n\n{content}\n" if not content.startswith("#") else content,
                encoding="utf-8",
            )
            tracker.log("learning_recorded", title=title, filename=filename)
            return ToolResult(True, json.dumps({"filename": filename}))
        except Exception as exc:
            return ToolResult(False, f"error: {exc}")

    # ------------------------------------------------------------------ #
    # update_memory                                                         #
    # ------------------------------------------------------------------ #
    def update_memory(content: str) -> ToolResult:
        if "update_memory" not in allowed_actions:
            return ToolResult(False, f"error: update_memory is not enabled for evolution surface '{surface}'")
        try:
            path = guard.resolve("MEMORY.md")
            path.write_text(content if content.endswith("\n") else content + "\n", encoding="utf-8")
            tracker.log("memory_updated")
            return ToolResult(True, "MEMORY.md updated")
        except Exception as exc:
            return ToolResult(False, f"error: {exc}")

    # ------------------------------------------------------------------ #
    # create_skill                                                          #
    # ------------------------------------------------------------------ #
    def create_skill(skill_name: str, description: str, script_content: str) -> ToolResult:
        if "create_skill" not in allowed_actions:
            return ToolResult(False, f"error: create_skill is not enabled for evolution surface '{surface}'")
        try:
            safe = _safe_name(skill_name)
            skill_dir = guard.resolve(f"skills/{safe}")
            skill_dir.mkdir(parents=True, exist_ok=True)
            (skill_dir / "SKILL.md").write_text(
                f"# {skill_name}\n\n{description}\n", encoding="utf-8"
            )
            (skill_dir / "main.py").write_text(
                script_content if script_content.endswith("\n") else script_content + "\n",
                encoding="utf-8",
            )
            tracker.log("skill_created", skill_name=safe)
            return ToolResult(
                True,
                json.dumps({"skill_name": safe, "path": f"skills/{safe}/main.py"}),
            )
        except Exception as exc:
            return ToolResult(False, f"error: {exc}")

    # ------------------------------------------------------------------ #
    # create_tool                                                           #
    # ------------------------------------------------------------------ #
    def create_tool(
        tool_name: str,
        description: str,
        parameters_schema: dict[str, Any],
        script_content: str,
    ) -> ToolResult:
        if "create_tool" not in allowed_actions:
            return ToolResult(False, f"error: create_tool is not enabled for evolution surface '{surface}'")
        try:
            safe = _safe_name(tool_name)
            tool_dir = guard.resolve(f"created_tools/{safe}")
            tool_dir.mkdir(parents=True, exist_ok=True)
            manifest = {
                "name": safe,
                "description": description,
                "parameters_schema": (
                    parameters_schema if isinstance(parameters_schema, dict)
                    else {"type": "object", "properties": {}}
                ),
            }
            (tool_dir / "manifest.json").write_text(
                json.dumps(manifest, indent=2) + "\n",
                encoding="utf-8",
            )
            (tool_dir / "README.md").write_text(
                f"# {tool_name}\n\n{description}\n",
                encoding="utf-8",
            )
            (tool_dir / "main.py").write_text(
                script_content if script_content.endswith("\n") else script_content + "\n",
                encoding="utf-8",
            )

            if registry is not None:
                def _created_tool_handler(**kwargs) -> ToolResult:
                    try:
                        env = os.environ.copy()
                        env["WORKSPACE_DIR"] = str(workspace_dir)
                        result = subprocess.run(
                            [sys.executable, str(tool_dir / "main.py")],
                            input=json.dumps(kwargs),
                            capture_output=True,
                            text=True,
                            cwd=str(workspace_dir),
                            env=env,
                        )
                        success = result.returncode == 0
                        output = result.stdout.strip() if result.stdout.strip() else result.stderr.strip()
                        return ToolResult(success, output or f"returncode={result.returncode}")
                    except Exception as exc:
                        return ToolResult(False, f"error: {exc}")

                registry.register(
                    safe,
                    _created_tool_handler,
                    description or f"Agent-created tool '{safe}'.",
                    manifest["parameters_schema"],
                )

            _refresh_tool_registry_doc()
            tracker.log("tool_created", tool_name=safe)
            return ToolResult(
                True,
                json.dumps({"tool_name": safe, "path": f"created_tools/{safe}/main.py"}),
            )
        except Exception as exc:
            return ToolResult(False, f"error: {exc}")

    # ------------------------------------------------------------------ #
    # run_skill                                                             #
    # ------------------------------------------------------------------ #
    def run_skill(skill_name: str, args: list[str] | None = None) -> ToolResult:
        try:
            safe = _safe_name(skill_name)
            script = guard.resolve(f"skills/{safe}/main.py")
            if not script.exists():
                skills_dir = guard.resolve("skills")
                available = [
                    d.name
                    for d in skills_dir.iterdir()
                    if d.is_dir() and (d / "main.py").exists()
                ]
                return ToolResult(
                    False,
                    f"error: skill '{skill_name}' not found. Available: {available}",
                )
            cmd = [sys.executable, str(script)] + [str(a) for a in (args or [])]
            env = os.environ.copy()
            env["WORKSPACE_DIR"] = str(workspace_dir)
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env=env,
                cwd=str(workspace_dir),
            )
            tracker.log(
                "skill_run",
                skill_name=safe,
                returncode=result.returncode,
                success=result.returncode == 0,
            )
            return ToolResult(
                result.returncode == 0,
                json.dumps({
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "returncode": result.returncode,
                }),
            )
        except Exception as exc:
            return ToolResult(False, f"error: {exc}")

    # ------------------------------------------------------------------ #
    # list_skills                                                           #
    # ------------------------------------------------------------------ #
    def list_skills() -> ToolResult:
        try:
            skills_dir = guard.resolve("skills")
            skills = [
                {"name": d.name, "description": _read_readme(d)}
                for d in sorted(skills_dir.iterdir())
                if d.is_dir() and (d / "main.py").exists()
            ]
            return ToolResult(True, json.dumps(skills))
        except Exception as exc:
            return ToolResult(False, f"error: {exc}")

    # ------------------------------------------------------------------ #
    # list_created_tools                                                    #
    # ------------------------------------------------------------------ #
    def list_created_tools() -> ToolResult:
        try:
            created_dir = guard.resolve("created_tools")
            tools: list[dict[str, Any]] = []
            for tool_dir in sorted(path for path in created_dir.iterdir() if path.is_dir()):
                manifest_path = tool_dir / "manifest.json"
                if not manifest_path.exists():
                    continue
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                tools.append(
                    {
                        "name": manifest.get("name", tool_dir.name),
                        "description": manifest.get("description", ""),
                        "path": f"created_tools/{tool_dir.name}/main.py",
                    }
                )
            return ToolResult(True, json.dumps(tools))
        except Exception as exc:
            return ToolResult(False, f"error: {exc}")

    def _read_readme(skill_dir: Path) -> str:
        readme = skill_dir / "SKILL.md"
        if not readme.exists():
            readme = skill_dir / "README.md"
        if not readme.exists():
            return ""
        lines = readme.read_text(encoding="utf-8").splitlines()
        # Return first non-heading, non-empty line as the summary
        for line in lines:
            stripped = line.strip().lstrip("#").strip()
            if stripped:
                return stripped
        return ""

    # ------------------------------------------------------------------ #
    # list_learnings                                                        #
    # ------------------------------------------------------------------ #
    def list_learnings() -> ToolResult:
        try:
            ld = guard.resolve(".learnings")
            files = sorted(p.name for p in ld.iterdir() if p.is_file())
            return ToolResult(True, json.dumps(files))
        except Exception as exc:
            return ToolResult(False, f"error: {exc}")

    # ------------------------------------------------------------------ #
    # read_learning                                                         #
    # ------------------------------------------------------------------ #
    def read_learning(filename: str) -> ToolResult:
        try:
            path = guard.resolve(f".learnings/{Path(filename).name}")
            if not path.exists():
                return ToolResult(False, f"error: learning '{filename}' not found")
            return ToolResult(True, path.read_text(encoding="utf-8"))
        except Exception as exc:
            return ToolResult(False, f"error: {exc}")

    # ------------------------------------------------------------------ #
    # apply_controller_update                                               #
    # ------------------------------------------------------------------ #
    def apply_controller_update(target: str, content: str, reason: str = "") -> ToolResult:
        """
        Propose a rewrite of one of the controller files (AGENTS.md, SOUL.md, TOOLS.md,
        MEMORY.md). The proposal is run through the gate before any
        write occurs — forbidden markers, suspicious phrases, diff size, and
        token expansion are all checked.
        """
        try:
            if "apply_controller_update" not in allowed_actions:
                return ToolResult(False, f"error: apply_controller_update is not enabled for evolution surface '{surface}'")

            if target not in writable_targets:
                allowed = ", ".join(sorted(writable_targets))
                return ToolResult(False, f"error: '{target}' is not an allowed controller file. Allowed: {allowed}")

            path = guard.resolve(target)
            before_text = path.read_text(encoding="utf-8") if path.exists() else ""

            proposal = UpdateProposal(
                target=target,
                reason=reason or "agent_authored_self_update",
                before_text=before_text,
                after_text=content,
            )
            accepted, gate_reason = apply_gate(proposal)

            if accepted:
                path.write_text(
                    content if content.endswith("\n") else content + "\n",
                    encoding="utf-8",
                )
                tracker.log(
                    "controller_update_applied",
                    target=target,
                    reason=reason,
                    gate_reason=gate_reason,
                )
            else:
                tracker.log(
                    "controller_update_rejected",
                    target=target,
                    reason=reason,
                    gate_reason=gate_reason,
                )

            return ToolResult(
                accepted,
                json.dumps({
                    "accepted": accepted,
                    "target": target,
                    "gate_reason": gate_reason,
                }),
            )
        except Exception as exc:
            return ToolResult(False, f"error: {exc}")

    # ------------------------------------------------------------------ #
    # Tool specs                                                            #
    # ------------------------------------------------------------------ #
    tool_specs: list[tuple[str, Any, str, dict]] = [
        (
            "record_learning",
            record_learning,
            (
                "Save a reflection note to .learnings/ as an audit log for self-evolution decisions. "
                "This records what you learned, but it is not the durable behavior mechanism; "
                "promote it into an enabled persistent update when it should affect future behavior, "
                "or treat it as ignorable/discardable audit residue."
            ),
            {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Short title for the learning."},
                    "content": {"type": "string", "description": "The learning content (markdown)."},
                },
                "required": ["title", "content"],
            },
        ),
    ]

    if "update_memory" in allowed_actions:
        tool_specs.append(
            (
                "update_memory",
                update_memory,
                (
                    "Rewrite your MEMORY.md file — your persistent scratchpad that survives across all tasks. "
                    "Use this to track durable long-term-memory summaries and important persistent context."
                ),
                {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "Full new content for MEMORY.md (replaces existing content).",
                        }
                    },
                    "required": ["content"],
                },
            )
        )

    if "create_skill" in allowed_actions:
        tool_specs.append(
            (
                "create_skill",
                create_skill,
                (
                    "Write a reusable Python script to skills/<skill_name>/main.py. "
                    "Also create skills/<skill_name>/SKILL.md describing when to use it. "
                    "The script can access the workspace via os.environ['WORKSPACE_DIR']. "
                    "It receives any arguments you pass via sys.argv[1:]. "
                    "Print results to stdout — that output is returned to you when you run the skill."
                ),
                {
                    "type": "object",
                    "properties": {
                        "skill_name": {
                            "type": "string",
                            "description": "Short name for the skill (used as the directory name).",
                        },
                        "description": {
                            "type": "string",
                            "description": "What the skill does and how to use it.",
                        },
                        "script_content": {
                            "type": "string",
                            "description": "Complete Python source code for the skill script.",
                        },
                    },
                    "required": ["skill_name", "description", "script_content"],
                },
            )
        )

    if "create_tool" in allowed_actions:
        tool_specs.append(
            (
                "create_tool",
                create_tool,
                (
                    "Create a reusable callable Python tool under created_tools/<tool_name>/. "
                    "The script receives JSON arguments on stdin and should print its result to stdout. "
                    "Use this when you want a new direct callable capability in addition to skills."
                ),
                {
                    "type": "object",
                    "properties": {
                        "tool_name": {
                            "type": "string",
                            "description": "Short name for the tool (used as the directory and tool name).",
                        },
                        "description": {
                            "type": "string",
                            "description": "What the tool does and when to use it.",
                        },
                        "parameters_schema": {
                            "type": "object",
                            "description": "JSON Schema describing the tool inputs.",
                        },
                        "script_content": {
                            "type": "string",
                            "description": (
                                "Complete Python source code for the tool. Read JSON args from stdin "
                                "and print the result to stdout."
                            ),
                        },
                    },
                    "required": ["tool_name", "description", "parameters_schema", "script_content"],
                },
            )
        )

    tool_specs.extend([
        (
            "run_skill",
            run_skill,
            (
                "Execute a previously created skill script and return its stdout. "
                "The script receives WORKSPACE_DIR as an environment variable."
            ),
            {
                "type": "object",
                "properties": {
                    "skill_name": {
                        "type": "string",
                        "description": "Name of the skill to run (must match a name from list_skills).",
                    },
                    "args": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional command-line arguments to pass to the script.",
                    },
                },
                "required": ["skill_name"],
            },
        ),
        (
            "list_skills",
            list_skills,
            "List all skills you have created, with their descriptions.",
            {"type": "object", "properties": {}},
        ),
        (
            "list_created_tools",
            list_created_tools,
            "List all callable tools you have created under created_tools/.",
            {"type": "object", "properties": {}},
        ),
        (
            "list_learnings",
            list_learnings,
            "List all saved reflection/audit notes in .learnings/.",
            {"type": "object", "properties": {}},
        ),
        (
            "read_learning",
            read_learning,
            "Read the content of a specific reflection/audit note.",
            {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Filename from list_learnings.",
                    }
                },
                "required": ["filename"],
            },
        ),
    ])

    if "apply_controller_update" in allowed_actions:
        tool_specs.append(
            (
                "apply_controller_update",
                apply_controller_update,
                (
                    "Propose a rewrite of one of your controller files "
                    f"({', '.join(writable_targets)}). "
                    "The update is gated: forbidden markers, unsafe phrases, "
                    "large diffs, and token expansion are all rejected. "
                    "Use this to update only the controller files permitted by the active evolution surface. "
                    "The new content replaces the file entirely."
                ),
                {
                    "type": "object",
                    "properties": {
                        "target": {
                            "type": "string",
                            "enum": list(writable_targets),
                            "description": "Which controller file to update.",
                        },
                        "content": {
                            "type": "string",
                            "description": "Full new content for the file.",
                        },
                        "reason": {
                            "type": "string",
                            "description": "Brief explanation of why this update is being made.",
                        },
                    },
                    "required": ["target", "content"],
                },
            )
        )

    return tool_specs
