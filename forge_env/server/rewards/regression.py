from forge_env.server.config import GAMMA


def regression_penalty(newly_failing_tasks: int) -> float:
    return GAMMA * float(max(0, newly_failing_tasks))
