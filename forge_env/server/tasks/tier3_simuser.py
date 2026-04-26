"""T3: lightweight simulated user / hidden-goals tasks (PRD Appendix A style)."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List


@dataclass
class Tier3Task:
    task_id: str
    prompt: str
    tier: str = "T3"
    task_type: str = "shopping"
    hidden_cues: List[str] = field(default_factory=lambda: ["32gb", "amd", "battery"])
    template_family: str = "alpha"


def generate_tier3(seed: int, count: int, family: str = "alpha") -> List[Tier3Task]:
    rng = random.Random(seed)
    variants = [
        (
            "Help the user pick a laptop under $1500 for video editing. Ask concise questions.",
            ["32gb", "amd", "battery"],
        ),
        (
            "A picky shopper wants noise-cancelling headphones under $300. Discover priorities.",
            ["comfort", "warranty", "usb-c"],
        ),
    ]
    tasks: List[Tier3Task] = []
    for idx in range(count):
        prompt, cues = rng.choice(variants)
        if family == "beta":
            cues = list(reversed(cues))
        tasks.append(
            Tier3Task(
                task_id=f"t3_{family}_{seed}_{idx}",
                prompt=prompt,
                hidden_cues=cues,
                template_family=family,
            )
        )
    return tasks


def tier3_task_to_dict(t: Tier3Task) -> dict:
    return {
        "task_id": t.task_id,
        "prompt": t.prompt,
        "tier": t.tier,
        "task_type": t.task_type,
        "hidden_cues": list(t.hidden_cues or []),
        "template_family": t.template_family,
    }


def grade_tier3(response: str, task: dict) -> float:
    """Soft match on hidden cues in the assistant reply."""
    lower = response.lower()
    cues = task.get("hidden_cues") or []
    if not cues:
        return 0.5
    hit = sum(1 for c in cues if c.lower() in lower)
    return hit / len(cues)
