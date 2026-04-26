from forge_env.models import ForgeAction
from forge_env.server.forge_environment import ForgeEnvironment


def test_reset_step_state_contract():
    env = ForgeEnvironment()
    obs = env.reset()
    assert obs.phase == "awaiting_edit"
    step_obs = env.step(ForgeAction(type="propose_edit", edit={"type": "no_op"}))
    assert step_obs.phase in {"edit_evaluated", "budget_violation"}
    st = env.state
    assert st.playbook_version >= 0
    assert st.curriculum_phase == "warmup"
