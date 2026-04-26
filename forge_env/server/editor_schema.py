"""Structured Editor output (PRD §4.2) — JSON parse + validation."""

from __future__ import annotations

import json
from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel


class EditPromptAction(BaseModel):
    type: Literal["edit_prompt"] = "edit_prompt"
    target: str
    new_content: str
    rationale: str = ""


class WriteLessonAction(BaseModel):
    type: Literal["write_lesson"] = "write_lesson"
    target: str = "lessons/general.md"
    content: str
    rationale: str = ""


class AddToolAction(BaseModel):
    type: Literal["add_tool"] = "add_tool"
    template: str = "summarise_text"
    tool_name: str = ""
    rationale: str = ""


class NoOpAction(BaseModel):
    type: Literal["no_op"] = "no_op"
    rationale: str = ""


class EditorOutput(BaseModel):
    diagnosis: str = ""
    action: Dict[str, Any]


def parse_editor_json(raw: str) -> Optional[EditorOutput]:
    try:
        data = json.loads(raw)
        return EditorOutput.model_validate(data)
    except Exception:
        return None


def editor_output_to_edit_dict(output: EditorOutput) -> Dict[str, Any]:
    """Flatten to the edit dict consumed by ``apply_edit``."""
    return dict(output.action)


def validate_action_shape(action: Dict[str, Any]) -> Dict[str, Any]:
    """Return a normalized edit dict or ``{\"type\": \"no_op\"}`` on failure."""
    t = action.get("type", "no_op")
    try:
        if t == "edit_prompt":
            m = EditPromptAction.model_validate(action)
            return m.model_dump()
        if t == "write_lesson":
            m = WriteLessonAction.model_validate(action)
            return m.model_dump()
        if t == "add_tool":
            m = AddToolAction.model_validate(action)
            return m.model_dump()
        if t == "no_op":
            m = NoOpAction.model_validate({**action, "type": "no_op"})
            return m.model_dump()
    except Exception:
        pass
    return {"type": "no_op", "rationale": "schema_validation_failed"}
