"""Run learned tools inside a throwaway Docker container.

Isolation (vs host):
- ``--network none``: no egress/ingress
- ``--read-only`` + ``--tmpfs /tmp``: writable temp only; package tree mounted read-only
- ``--cap-drop ALL`` + ``no-new-privileges``: reduce privilege escalation
- Docker's default seccomp profile applies (blocks many dangerous syscalls)

Requires a working Docker daemon. Set ``FORGE_TOOL_ISOLATION=subprocess`` to skip.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import forge_env

from forge_env.server.sandbox.isolation import tool_docker_image


def _workspace_root() -> str:
    override = os.environ.get("FORGE_WORKSPACE_ROOT")
    if override:
        return str(Path(override).resolve())
    pkg_dir = Path(forge_env.__file__).resolve().parent
    return str(pkg_dir.parent)


def run_tool_source_docker(
    source: str,
    args: dict[str, Any],
    timeout_sec: int = 5,
) -> tuple[int, str, Any | None]:
    payload = json.dumps({"code": source, "args": args})
    workspace = _workspace_root()
    image = tool_docker_image()
    # Allow time for first-time image pull (override with FORGE_DOCKER_RUN_TIMEOUT).
    run_timeout = int(
        os.environ.get(
            "FORGE_DOCKER_RUN_TIMEOUT",
            str(max(timeout_sec + 15, 120)),
        )
    )

    cmd = [
        "docker",
        "run",
        "--rm",
        "-i",
        "--network",
        "none",
        "--read-only",
        "--tmpfs",
        "/tmp:rw,size=32m,noexec,nosuid,nodev",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--pids-limit",
        "64",
        "--memory",
        "256m",
        "--cpus",
        "0.5",
        "-e",
        "PYTHONPATH=/workspace",
        "-e",
        "PYTHONUNBUFFERED=1",
        "-v",
        f"{workspace}:/workspace:ro",
        "-w",
        "/workspace",
        image,
        "python",
        "-m",
        "forge_env.server.sandbox.run_tool",
    ]

    result = subprocess.run(
        cmd,
        input=payload,
        capture_output=True,
        text=True,
        timeout=run_timeout,
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
