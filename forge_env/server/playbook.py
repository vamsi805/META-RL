from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import yaml


@dataclass
class Playbook:
    root: Path
    version: int
    parent: int | None
    prompts: Dict[str, str]
    lessons: Dict[str, str]
    routing: Dict[str, Any]
    tools_registry: Dict[str, Any]
    learned_tools: Dict[str, str] = field(default_factory=dict)
    edit_log: List[Dict[str, Any]] = field(default_factory=list)

    def copy(self) -> "Playbook":
        return copy.deepcopy(self)

    def token_count(self) -> int:
        from forge_env.server.tokenizer import count_tokens_playbook

        return count_tokens_playbook(self)

    def tool_count(self) -> int:
        builtin = len(self.tools_registry.get("tools", []))
        return builtin + len(self.learned_tools)

    def snapshot(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "parent": self.parent,
            "prompts": self.prompts,
            "lessons": self.lessons,
            "routing": self.routing,
            "tool_count": self.tool_count(),
            "token_count": self.token_count(),
            "edit_log": self.edit_log[-10:],
        }


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def load_initial_playbook(root: Path) -> Playbook:
    playbook_dir = root / "playbook"
    manifest = yaml.safe_load(_read_text(playbook_dir / "manifest.yaml")) or {}
    prompts_dir = playbook_dir / "prompts"
    lessons_dir = playbook_dir / "lessons"

    prompts = {
        p.stem: _read_text(p)
        for p in prompts_dir.glob("*.md")
    }

    lessons = {"general": _read_text(lessons_dir / "general.md")}
    per_domain = lessons_dir / "per_domain"
    if per_domain.exists():
        for p in per_domain.glob("*.md"):
            lessons[f"per_domain/{p.stem}"] = _read_text(p)

    routing = yaml.safe_load(_read_text(playbook_dir / "routing.yaml")) or {"routes": {}}
    tools_registry = yaml.safe_load(_read_text(playbook_dir / "tools" / "_registry.yaml")) or {"tools": []}

    return Playbook(
        root=playbook_dir,
        version=int(manifest.get("version", 0)),
        parent=manifest.get("parent"),
        prompts=prompts,
        lessons=lessons,
        routing=routing,
        tools_registry=tools_registry,
        edit_log=[],
    )
