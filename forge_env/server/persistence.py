from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

from forge_env.server.playbook import Playbook


def write_playbook_checkpoint(
    playbook: Playbook,
    reward_components: Dict[str, float] | None,
    base_dir: Path | None = None,
) -> Path:
    root = base_dir or Path("outputs") / "playbook_versions"
    root.mkdir(parents=True, exist_ok=True)
    vdir = root / f"v{playbook.version}"
    vdir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "version": playbook.version,
        "parent": playbook.parent,
        "reward_components": reward_components or {},
        "metrics_snapshot": {
            "prompt_token_count": playbook.token_count(),
            "tool_count": playbook.tool_count(),
            "lesson_count": len([k for k in playbook.lessons if k]),
        },
        "snapshot": playbook.snapshot(),
    }
    path = vdir / "manifest.yaml"
    path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    return path
