from forge_env.server.editor_schema import parse_editor_json, validate_action_shape


def test_validate_action_shape_no_op():
    assert validate_action_shape({"type": "no_op"})["type"] == "no_op"


def test_parse_editor_json():
    raw = '{"diagnosis":"x","action":{"type":"no_op"}}'
    out = parse_editor_json(raw)
    assert out is not None
    assert out.action["type"] == "no_op"


def test_invalid_action_becomes_no_op():
    assert validate_action_shape({"type": "edit_prompt"})["type"] == "no_op"
