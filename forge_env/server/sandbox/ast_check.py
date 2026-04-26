"""Static analysis for learned tools: allow-list imports + banned constructs.

This is defense-in-depth alongside restricted builtins in run_tool.py.
It is not a full proof of isolation; run untrusted code only inside Docker
or a dedicated VM, and use resource limits on the tool subprocess.
"""

from __future__ import annotations

import ast
from typing import FrozenSet

ALLOWED_TOP_LEVEL_IMPORTS: FrozenSet[str] = frozenset(
    {
        "re",
        "math",
        "json",
        "string",
        "datetime",
        "typing",
        "collections",
        "itertools",
        "functools",
        "decimal",
        "fractions",
    }
)

BANNED_CALL_NAMES: FrozenSet[str] = frozenset(
    {
        "open",
        "exec",
        "eval",
        "compile",
        "__import__",
        "getattr",
        "setattr",
        "delattr",
        "globals",
        "locals",
        "vars",
        "dir",
        "input",
        "breakpoint",
        "help",
        "memoryview",
    }
)


def _import_root(name: str | None) -> str | None:
    if not name:
        return None
    return name.split(".", 1)[0]


def is_safe_python(source: str) -> bool:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = _import_root(alias.name)
                if root is None or root not in ALLOWED_TOP_LEVEL_IMPORTS:
                    return False
        elif isinstance(node, ast.ImportFrom):
            root = _import_root(node.module)
            if root is None or root not in ALLOWED_TOP_LEVEL_IMPORTS:
                return False
        elif isinstance(
            node,
            (
                ast.ClassDef,
                ast.AsyncFunctionDef,
                ast.With,
                ast.AsyncWith,
                ast.Try,
                ast.TryStar,
                ast.Raise,
                ast.Global,
                ast.Nonlocal,
                ast.Delete,
            ),
        ):
            return False
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id in BANNED_CALL_NAMES or node.func.id.startswith("_"):
                    return False
            if isinstance(node.func, ast.Attribute):
                if isinstance(node.func.attr, str) and (
                    node.func.attr.startswith("__") or node.func.attr.endswith("__")
                ):
                    return False
        elif isinstance(node, ast.Attribute):
            if isinstance(node.attr, str) and (
                node.attr.startswith("__") or node.attr.endswith("__")
            ):
                return False
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if node.id in BANNED_CALL_NAMES:
                return False

    return True
