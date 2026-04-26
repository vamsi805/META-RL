from __future__ import annotations

import json
import subprocess
import sys
from typing import Any

from forge_env.server.sandbox.docker_runner import run_tool_source_docker
from forge_env.server.sandbox.isolation import ToolIsolationMode, resolve_tool_isolation_mode


def _run_tool_subprocess(
    source: str,
    args: dict[str, Any],
    timeout_sec: int = 5,
) -> tuple[int, str, Any | None]:
    payload = json.dumps({"code": source, "args": args})
    result = subprocess.run(
        [sys.executable, "-m", "forge_env.server.sandbox.run_tool"],
        input=payload,
        capture_output=True,
        text=True,
        timeout=timeout_sec,
        check=False,
    )
    log = (result.stdout or "") + (result.stderr or "")
    if result.returncode != 0:
        return result.returncode, log, None
    try:
        line = (result.stdout or "").strip().splitlines()[-1]
        data = json.loads(line)
        if data.get("ok"):
            return 0, log, data.get("result")
    except (json.JSONDecodeError, IndexError):
        pass
    return 1, log, None


def run_tool_source(
    source: str,
    args: dict[str, Any],
    timeout_sec: int = 5,
) -> tuple[int, str, Any | None]:
    """Execute tool via Docker (preferred) or local subprocess (fallback / dev)."""
    mode = resolve_tool_isolation_mode()
    if mode is ToolIsolationMode.DOCKER:
        return run_tool_source_docker(source, args, timeout_sec=timeout_sec)
    return _run_tool_subprocess(source, args, timeout_sec=timeout_sec)
