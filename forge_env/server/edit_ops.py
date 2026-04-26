from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict

from forge_env.server.playbook import Playbook
from forge_env.server.sandbox.ast_check import is_safe_python
from forge_env.server.sandbox.subprocess_runner import run_tool_source


class ToolValidationError(Exception):
    """Raised when a learned tool fails AST or smoke validation."""


TOOL_TEMPLATES: Dict[str, str] = {
    "summarise_text": """def run(text: str, style: str = "concise") -> dict:
    words = text.split()
    if style == "bullet":
        return {"summary": "- " + " ".join(words[:30])}
    return {"summary": " ".join(words[:40])}
""",
    "extract_pattern": """import re

def run(text: str, regex: str) -> dict:
    matches = re.findall(regex, text)
    return {"matches": matches}
""",
}

TEMPLATE_SMOKE_ARGS: Dict[str, Dict[str, Any]] = {
    "summarise_text": {"text": "hello world test", "style": "concise"},
    "extract_pattern": {"text": "abc123", "regex": r"\d+"},
}


def _validate_template_tool(template_name: str, source: str) -> None:
    if not is_safe_python(source):
        raise ToolValidationError("ast_reject")
    smoke = TEMPLATE_SMOKE_ARGS.get(template_name)
    if smoke is None:
        raise ToolValidationError("unknown_template")
    rc, log, result = run_tool_source(source, smoke, timeout_sec=5)
    if rc != 0 or result is None:
        raise ToolValidationError(f"smoke_failed:{log[:400]}")


def apply_edit(playbook: Playbook, action: Dict[str, Any]) -> Playbook:
    pb = playbook.copy()
    action_type = action.get("type", "no_op")

    if action_type == "edit_prompt":
        target = action.get("target", "")
        name = target.replace("prompts/", "").replace(".md", "")
        new_content = action.get("new_content", "")
        pb.prompts[name] = new_content
    elif action_type == "write_lesson":
        target = action.get("target", "lessons/general.md")
        lesson_key = (
            "general"
            if "general.md" in target
            else f"per_domain/{target.split('/')[-1].replace('.md', '')}"
        )
        old = pb.lessons.get(lesson_key, "")
        pb.lessons[lesson_key] = (old + "\n" + action.get("content", "")).strip()
    elif action_type == "add_tool":
        template_name = action.get("template", "summarise_text")
        tool_name = action.get("tool_name", f"learned_tool_{pb.version + 1}")
        template = TOOL_TEMPLATES.get(template_name)
        if not template:
            raise ToolValidationError("unknown_template_name")
        _validate_template_tool(template_name, template)
        pb.learned_tools[f"{tool_name}.py"] = template
    elif action_type == "no_op":
        pass
    else:
        pass

    pb.parent = playbook.version
    pb.version = playbook.version + 1
    pb.edit_log.append(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": action_type,
            "target": action.get("target"),
            "rationale": action.get("rationale", ""),
        }
    )
    return pb
