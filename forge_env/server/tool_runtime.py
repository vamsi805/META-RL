from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from forge_env.server.playbook import Playbook
from forge_env.server.sandbox.subprocess_runner import run_tool_source


@dataclass
class ToolRunResult:
    ok: bool
    result: Any | None
    error: str = ""


class ToolRuntime:
    """Runs built-in and learned playbook tools through the sandbox runner."""

    def __init__(self, timeout_sec: int = 8) -> None:
        self.timeout_sec = timeout_sec

    def available_tools(self, playbook: Playbook, task_type: str) -> list[dict[str, Any]]:
        route = (playbook.routing.get("routes", {}) or {}).get(task_type, {})
        route_names = set(route.get("tools", []))
        tools = []
        for item in playbook.tools_registry.get("tools", []):
            if not item.get("enabled", True):
                continue
            name = item.get("name")
            if route_names and name not in route_names:
                continue
            tools.append({"name": name, "source": "builtin", "path": item.get("path")})
        for filename in sorted(playbook.learned_tools):
            name = filename.removesuffix(".py")
            tools.append({"name": name, "source": "learned", "path": filename})
        return tools

    def run_tool(
        self,
        playbook: Playbook,
        name: str,
        args: Dict[str, Any],
    ) -> ToolRunResult:
        source = self._source_for_tool(playbook, name)
        if source is None:
            return ToolRunResult(False, None, f"unknown_tool:{name}")
        rc, log, result = run_tool_source(source, args, timeout_sec=self.timeout_sec)
        if rc != 0:
            return ToolRunResult(False, None, log[:800])
        return ToolRunResult(True, result)

    def _source_for_tool(self, playbook: Playbook, name: str) -> str | None:
        learned_name = f"{name}.py"
        if learned_name in playbook.learned_tools:
            return playbook.learned_tools[learned_name]

        for item in playbook.tools_registry.get("tools", []):
            if item.get("name") != name:
                continue
            rel_path = item.get("path")
            if not rel_path:
                return None
            path = playbook.root / rel_path
            if not path.exists():
                return None
            return path.read_text(encoding="utf-8")
        return None
