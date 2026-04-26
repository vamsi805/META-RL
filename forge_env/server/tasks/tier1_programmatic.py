from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List

# Train/val use family "alpha"; holdout uses "beta" (unseen wording) for generalisation.
NAMES_ALPHA = ["Alice", "Bob", "Carol", "Deepa", "Ethan", "Farah"]
SLOTS_ALPHA = ["Mon 10:00", "Tue 14:00", "Wed 13:00", "Thu 11:00", "Fri 09:00"]

NAMES_BETA = ["Gina", "Hassan", "Iris", "Jun", "Kara", "Leo"]
SLOTS_BETA = ["Sat 09:00", "Sat 15:00", "Sun 11:00", "Sun 16:00"]


@dataclass
class Tier1Task:
    task_id: str
    prompt: str
    expected: Dict[str, str]
    tier: str = "T1"
    task_type: str = "scheduling"
    template_family: str = "alpha"


def generate_tier1(
    seed: int,
    count: int,
    family: str = "alpha",
) -> List[Tier1Task]:
    rng = random.Random(seed)
    if family == "beta":
        names, slots = NAMES_BETA, SLOTS_BETA
        duration = "45"
        verb = "Book a 45-minute planning session"
    else:
        names, slots = NAMES_ALPHA, SLOTS_ALPHA
        duration = "60"
        verb = "Schedule a 60-minute meeting"

    tasks: List[Tier1Task] = []
    for idx in range(count):
        people = rng.sample(names, k=3)
        slot = rng.choice(slots)
        prompt = f"{verb} for {', '.join(people)}. Find earliest valid slot."
        expected = {"slot": slot, "duration": duration}
        tasks.append(
            Tier1Task(
                task_id=f"t1_{family}_{seed}_{idx}",
                prompt=prompt,
                expected=expected,
                template_family=family,
            )
        )
    return tasks


def tier1_task_to_dict(t: Tier1Task) -> dict:
    return {
        "task_id": t.task_id,
        "prompt": t.prompt,
        "expected": t.expected,
        "tier": t.tier,
        "task_type": t.task_type,
        "template_family": t.template_family,
    }


def grade_tier1(prediction: Dict[str, str], expected: Dict[str, str]) -> float:
    return 1.0 if prediction == expected else 0.0
