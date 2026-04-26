from __future__ import annotations

import json
from typing import Any, Dict, List

from forge_env.server.editor_schema import (
    editor_output_to_edit_dict,
    parse_editor_json,
    validate_action_shape,
)
from forge_env.server.llm import extract_json_object, get_configured_llm_client
from forge_env.server.playbook import Playbook


def stub_candidate_edits(step: int, k: int) -> List[Dict[str, Any]]:
    templates: List[Dict[str, Any]] = [
        {
            "type": "edit_prompt",
            "target": "prompts/reflector.md",
            "new_content": (
                "Before final answer, re-read the task and list every constraint. "
                "For scheduling tasks, mention earliest slot explicitly.\n"
                f"checkpoint={step}"
            ),
            "rationale": "Improve constraint coverage",
        },
        {
            "type": "write_lesson",
            "target": "lessons/general.md",
            "content": (
                f"Lesson {step}: for scheduling, always return the true earliest slot; "
                "for email, acknowledge, explain cause, propose plan."
            ),
            "rationale": "Distill cross-task heuristic",
        },
        {
            "type": "add_tool",
            "template": "summarise_text",
            "tool_name": f"summary_tool_{step}",
            "rationale": "Composable summary helper",
        },
        {
            "type": "edit_prompt",
            "target": "prompts/executor.md",
            "new_content": (
                "Take one action at a time. Prefer tools that resolve uncertainty. "
                "For email tasks, produce a full professional reply.\n"
                f"revision={step}"
            ),
            "rationale": "Executor clarity",
        },
        {
            "type": "write_lesson",
            "target": "lessons/per_domain/scheduling.md",
            "content": "Earliest valid slot must match task participants and duration.",
            "rationale": "Domain scheduling",
        },
        {"type": "no_op", "rationale": "No change"},
        {
            "type": "edit_prompt",
            "target": "prompts/planner.md",
            "new_content": (
                "Decompose into sub-goals; track constraints; verify before final output.\n"
                f"step={step}"
            ),
            "rationale": "Planner tightening",
        },
        {
            "type": "add_tool",
            "template": "extract_pattern",
            "tool_name": f"extract_tool_{step}",
            "rationale": "Regex helper",
        },
    ]
    return [validate_action_shape(templates[(step + i) % len(templates)]) for i in range(k)]


def llm_candidate_edits(
    *,
    step: int,
    k: int,
    playbook: Playbook,
    failures_buffer: List[Dict[str, Any]],
    val_baseline: float,
    backend: str | None = None,
    model_name: str | None = None,
) -> List[Dict[str, Any]]:
    client = get_configured_llm_client(backend=backend, model_name=model_name)
    if client is None:
        return stub_candidate_edits(step, k)

    prompt = {
        "step": step,
        "number_of_edits_needed": k,
        "current_validation_score": val_baseline,
        "playbook_snapshot": playbook.snapshot(),
        "recent_failures": failures_buffer[:6],
        "allowed_actions": [
            {
                "type": "edit_prompt",
                "fields": ["target", "new_content", "rationale"],
                "targets": [
                    "prompts/planner.md",
                    "prompts/executor.md",
                    "prompts/reflector.md",
                    "prompts/formatter.md",
                    "prompts/persona.md",
                ],
            },
            {
                "type": "write_lesson",
                "fields": ["target", "content", "rationale"],
                "targets": ["lessons/general.md", "lessons/per_domain/scheduling.md"],
            },
            {
                "type": "add_tool",
                "fields": ["template", "tool_name", "rationale"],
                "templates": ["summarise_text", "extract_pattern"],
            },
            {"type": "no_op", "fields": ["rationale"]},
        ],
    }
    messages = [
        {
            "role": "system",
            "content": (
                "You are the Forge editor. Propose playbook edits that may improve task "
                "scores. Return only JSON with this shape: "
                '{"candidates":[{"diagnosis":"...","action":{...}}]}. '
                "Do not return markdown."
            ),
        },
        {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
    ]
    raw = client.generate(messages, max_new_tokens=1200, temperature=0.7)
    data = extract_json_object(raw)
    candidates: List[Dict[str, Any]] = []
    if isinstance(data, dict):
        for item in data.get("candidates", []):
            parsed = parse_editor_json(json.dumps(item))
            if parsed is None:
                continue
            candidates.append(validate_action_shape(editor_output_to_edit_dict(parsed)))

    if not candidates:
        return stub_candidate_edits(step, k)

    while len(candidates) < k:
        candidates.append(stub_candidate_edits(step + len(candidates), 1)[0])
    return candidates[:k]
