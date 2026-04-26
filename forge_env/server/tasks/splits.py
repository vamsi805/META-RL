from __future__ import annotations

from typing import Dict, List

from forge_env.server.config import (
    HOLDOUT_T1,
    HOLDOUT_T2,
    HOLDOUT_T3,
    TRAIN_T1,
    TRAIN_T2,
    TRAIN_T3,
    VAL_T1,
    VAL_T2,
    VAL_T3,
)
from forge_env.server.tasks.tier1_programmatic import generate_tier1, tier1_task_to_dict
from forge_env.server.tasks.tier2_rubric import generate_tier2, tier2_task_to_dict
from forge_env.server.tasks.tier3_simuser import generate_tier3, tier3_task_to_dict


def build_task_splits(base_seed: int = 42) -> Dict[str, List[dict]]:
    """Train/val use template family *alpha*; holdout uses *beta* (unseen templates)."""
    s = base_seed
    train = (
        [tier1_task_to_dict(t) for t in generate_tier1(s + 1, TRAIN_T1, family="alpha")]
        + [tier2_task_to_dict(t) for t in generate_tier2(s + 2, TRAIN_T2, family="alpha")]
        + [tier3_task_to_dict(t) for t in generate_tier3(s + 3, TRAIN_T3, family="alpha")]
    )
    val = (
        [tier1_task_to_dict(t) for t in generate_tier1(s + 101, VAL_T1, family="alpha")]
        + [tier2_task_to_dict(t) for t in generate_tier2(s + 102, VAL_T2, family="alpha")]
        + [tier3_task_to_dict(t) for t in generate_tier3(s + 103, VAL_T3, family="alpha")]
    )
    holdout = (
        [tier1_task_to_dict(t) for t in generate_tier1(s + 201, HOLDOUT_T1, family="beta")]
        + [tier2_task_to_dict(t) for t in generate_tier2(s + 202, HOLDOUT_T2, family="beta")]
        + [tier3_task_to_dict(t) for t in generate_tier3(s + 203, HOLDOUT_T3, family="beta")]
    )
    return {"train": train, "val": val, "holdout": holdout}
