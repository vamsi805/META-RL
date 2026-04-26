from pathlib import Path

from forge_env.server.edit_ops import apply_edit
from forge_env.server.playbook import load_initial_playbook
from forge_env.server.rewards.compose import compute_reward


def test_reward_components_have_total():
    pb = load_initial_playbook(Path("forge_env"))
    new_pb = apply_edit(pb, {"type": "write_lesson", "content": "Check constraints first."})
    reward, components = compute_reward(pb, new_pb, baseline_val=0.4, new_val=0.5)
    assert isinstance(reward, float)
    assert "total" in components
