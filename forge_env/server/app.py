from openenv.core.env_server import create_fastapi_app

from forge_env.models import ForgeAction, ForgeObservation
from forge_env.server.forge_environment import ForgeEnvironment


def forge_env_factory() -> ForgeEnvironment:
    return ForgeEnvironment()


app = create_fastapi_app(forge_env_factory, ForgeAction, ForgeObservation)
