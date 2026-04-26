"""Select how learned tools are executed: Docker (full isolation) vs local subprocess."""

from __future__ import annotations

import os
import shutil
import subprocess
from enum import Enum
from functools import lru_cache


class ToolIsolationMode(str, Enum):
    AUTO = "auto"
    DOCKER = "docker"
    SUBPROCESS = "subprocess"


def _env_mode() -> str:
    return os.environ.get("FORGE_TOOL_ISOLATION", "auto").strip().lower()


@lru_cache(maxsize=1)
def docker_daemon_reachable() -> bool:
    if not shutil.which("docker"):
        return False
    try:
        r = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            timeout=5,
            check=False,
        )
        return r.returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return False


def resolve_tool_isolation_mode() -> ToolIsolationMode:
    raw = _env_mode()
    if raw == ToolIsolationMode.SUBPROCESS.value:
        return ToolIsolationMode.SUBPROCESS
    if raw == ToolIsolationMode.DOCKER.value:
        return ToolIsolationMode.DOCKER
    if raw == ToolIsolationMode.AUTO.value or not raw:
        if docker_daemon_reachable():
            return ToolIsolationMode.DOCKER
        return ToolIsolationMode.SUBPROCESS
    return ToolIsolationMode.SUBPROCESS


def tool_docker_image() -> str:
    return os.environ.get("FORGE_TOOL_DOCKER_IMAGE", "python:3.11-slim")
