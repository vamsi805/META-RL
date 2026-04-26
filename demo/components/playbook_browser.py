from __future__ import annotations

import difflib
from pathlib import Path

import yaml


def list_playbook_versions() -> list[str]:
    root = Path("outputs/playbook_versions")
    if not root.exists():
        return []
    return sorted(d.name for d in root.iterdir() if d.is_dir())


def load_manifest(version_dir: str) -> str:
    p = Path("outputs/playbook_versions") / version_dir / "manifest.yaml"
    if not p.exists():
        return ""
    return p.read_text(encoding="utf-8")


def prompt_diff_v0_vs_latest() -> str:
    """Unified diff of reflector prompt: disk v0 vs latest snapshot in checkpoints."""
    v0 = Path("forge_env/playbook/prompts/reflector.md")
    if not v0.exists():
        return "No v0 reflector on disk."
    v0_text = v0.read_text(encoding="utf-8").splitlines(keepends=True)
    versions = list_playbook_versions()
    if not versions:
        return "No checkpoints; run training to create outputs/playbook_versions/."
    latest = Path("outputs/playbook_versions") / versions[-1] / "manifest.yaml"
    data = yaml.safe_load(latest.read_text(encoding="utf-8"))
    snap = (data or {}).get("snapshot") or {}
    prompts = snap.get("prompts") or {}
    new_text = (prompts.get("reflector") or "").splitlines(keepends=True)
    diff = difflib.unified_diff(
        v0_text,
        new_text,
        fromfile="reflector_v0.md",
        tofile=f"reflector_{versions[-1]}.md",
    )
    return "".join(diff) or "(no textual change to reflector in snapshot)"


def metrics_tail_jsonl(n: int = 15) -> str:
    p = Path("outputs/smoke_metrics.jsonl")
    if not p.exists():
        return ""
    lines = p.read_text(encoding="utf-8").splitlines()
    return "\n".join(lines[-n:])
