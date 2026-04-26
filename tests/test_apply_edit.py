from pathlib import Path

from forge_env.server.edit_ops import apply_edit
from forge_env.server.playbook import load_initial_playbook


def test_edit_prompt_changes_content():
    pb = load_initial_playbook(Path("forge_env"))
    new_pb = apply_edit(
        pb,
        {
            "type": "edit_prompt",
            "target": "prompts/reflector.md",
            "new_content": "updated reflector",
        },
    )
    assert new_pb.prompts["reflector"] == "updated reflector"
    assert new_pb.version == pb.version + 1


def test_add_tool_from_template():
    pb = load_initial_playbook(Path("forge_env"))
    new_pb = apply_edit(pb, {"type": "add_tool", "template": "summarise_text", "tool_name": "x"})
    assert "x.py" in new_pb.learned_tools
