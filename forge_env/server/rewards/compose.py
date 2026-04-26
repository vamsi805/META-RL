from __future__ import annotations

from typing import Dict, Tuple

from forge_env.server.config import ALPHA, DELTA
from forge_env.server.playbook import Playbook
from forge_env.server.rewards.complexity import complexity_penalty
from forge_env.server.rewards.delta_validation import delta_validation
from forge_env.server.rewards.regression import regression_penalty


def compute_reward(
    old_pb: Playbook,
    new_pb: Playbook,
    baseline_val: float,
    new_val: float,
    newly_failing_tasks: int = 0,
    diversity_distance: float = 0.0,
) -> Tuple[float, Dict[str, float]]:
    dval = delta_validation(new_val, baseline_val)
    cpen = complexity_penalty(old_pb, new_pb)
    rpen = regression_penalty(newly_failing_tasks)
    dbonus = DELTA * max(0.0, diversity_distance)
    total = (ALPHA * dval) - cpen - rpen + dbonus
    return total, {
        "delta_val": dval,
        "complexity": -cpen,
        "regression": -rpen,
        "diversity": dbonus,
        "total": total,
    }
