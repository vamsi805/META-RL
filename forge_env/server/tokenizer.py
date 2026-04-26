"""Token counting for playbook budgets (tiktoken when available, else heuristic)."""

from __future__ import annotations

from typing import Any


def count_tokens_text(text: str) -> int:
    if not text:
        return 0
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return max(1, int(len(text.split()) * 4 // 3))


def count_tokens_playbook(playbook: Any) -> int:
    total = 0
    for text in playbook.prompts.values():
        total += count_tokens_text(text)
    for text in playbook.lessons.values():
        total += count_tokens_text(text)
    for text in playbook.learned_tools.values():
        total += count_tokens_text(text)
    return total
