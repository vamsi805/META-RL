from forge_env.server.config import BETA
from forge_env.server.playbook import Playbook


def complexity_penalty(old_pb: Playbook, new_pb: Playbook) -> float:
    token_delta = max(0, new_pb.token_count() - old_pb.token_count())
    return BETA * (token_delta / 100.0)
