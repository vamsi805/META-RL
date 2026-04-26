from forge_env.server.config import (
    LESSON_TOKEN_CAP,
    PROMPT_TOKEN_CAP,
    TOOL_COUNT_CAP,
    TOOL_LOC_CAP,
)
from forge_env.server.playbook import Playbook
from forge_env.server.tokenizer import count_tokens_text


def violates_budget(playbook: Playbook) -> bool:
    prompt_tokens = sum(count_tokens_text(v) for v in playbook.prompts.values())
    lesson_tokens = sum(
        count_tokens_text(v)
        for k, v in playbook.lessons.items()
        if k.startswith("general") or k.startswith("per_domain")
    )

    if prompt_tokens > PROMPT_TOKEN_CAP:
        return True
    if lesson_tokens > LESSON_TOKEN_CAP:
        return True
    if playbook.tool_count() > TOOL_COUNT_CAP:
        return True
    for tool_source in playbook.learned_tools.values():
        if len(tool_source.splitlines()) > TOOL_LOC_CAP:
            return True
    return False
