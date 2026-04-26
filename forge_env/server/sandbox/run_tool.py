"""Run learned tool source in a child process with tight resource limits.

stdin: JSON object {"code": str, "args": dict}
stdout: JSON object {"ok": true, "result": ...} on success
stderr: JSON object {"ok": false, "error": str} on failure

The parent should invoke this module via ``python -m forge_env.server.sandbox.run_tool``.
"""

from __future__ import annotations

import builtins
import json
import resource
import sys
import traceback
from typing import Any


def _linux_no_new_privs() -> None:
    """Best-effort hardening for the subprocess fallback path (Linux only)."""
    if sys.platform != "linux":
        return
    try:
        import ctypes

        libc = ctypes.CDLL(None)
        PR_SET_NO_NEW_PRIVS = 38
        libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0)
    except (AttributeError, OSError, TypeError):
        pass


def _restricted_builtins() -> dict[str, Any]:
    banned = frozenset(
        {
            "open",
            "compile",
            "eval",
            "exec",
            "__import__",
            "globals",
            "locals",
            "vars",
            "getattr",
            "setattr",
            "delattr",
            "input",
            "breakpoint",
            "help",
        }
    )
    out: dict[str, Any] = {}
    for name in dir(builtins):
        if name.startswith("_"):
            continue
        if name in banned:
            continue
        out[name] = getattr(builtins, name)
    return out


def _apply_limits() -> None:
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (5, 5))
    except (ValueError, OSError, AttributeError):
        pass
    try:
        limit = 256 * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    except (ValueError, OSError, AttributeError):
        pass
    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (48, 48))
    except (ValueError, OSError, AttributeError):
        pass
    try:
        resource.setrlimit(resource.RLIMIT_NPROC, (32, 32))
    except (ValueError, OSError, AttributeError):
        pass


def main() -> None:
    _linux_no_new_privs()
    _apply_limits()
    try:
        data = json.loads(sys.stdin.read())
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"ok": False, "error": f"invalid_input:{exc}"}), file=sys.stderr)
        sys.exit(2)

    code = data.get("code", "")
    args = data.get("args") or {}
    ns: dict[str, Any] = {}
    try:
        exec(compile(code, "<tool>", "exec"), {"__builtins__": _restricted_builtins()}, ns)
    except Exception:
        print(json.dumps({"ok": False, "error": traceback.format_exc()}), file=sys.stderr)
        sys.exit(1)

    run_fn = ns.get("run")
    if not callable(run_fn):
        print(json.dumps({"ok": False, "error": "missing_run_callable"}), file=sys.stderr)
        sys.exit(3)
    try:
        result = run_fn(**args)
    except Exception:
        print(json.dumps({"ok": False, "error": traceback.format_exc()}), file=sys.stderr)
        sys.exit(4)

    print(json.dumps({"ok": True, "result": result}))


if __name__ == "__main__":
    main()
