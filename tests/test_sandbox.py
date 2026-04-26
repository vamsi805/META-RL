from forge_env.server.sandbox.ast_check import is_safe_python
from forge_env.server.sandbox.subprocess_runner import run_tool_source


def test_ast_rejects_os_import():
    src = "import os\nprint('x')"
    assert not is_safe_python(src)


def test_ast_accepts_template_like_tool():
    src = """import re

def run(text: str, regex: str) -> dict:
    matches = re.findall(regex, text)
    return {"matches": matches}
"""
    assert is_safe_python(src)


def test_ast_rejects_dunder_attr():
    src = "x = (1).__class__"
    assert not is_safe_python(src)


def test_ast_rejects_importlib():
    src = "import importlib\nimportlib.import_module('os')"
    assert not is_safe_python(src)


def test_run_tool_subprocess_smoke():
    src = """def run(text: str, style: str = "concise") -> dict:
    return {"summary": text[:5]}
"""
    rc, _log, result = run_tool_source(src, {"text": "hello", "style": "concise"}, timeout_sec=5)
    assert rc == 0
    assert result == {"summary": "hello"}
