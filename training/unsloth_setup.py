from typing import Any


def maybe_patch_unsloth() -> tuple[bool, Any]:
    try:
        import unsloth  # type: ignore

        return True, unsloth
    except Exception:
        return False, None
