from __future__ import annotations

from typing import Any, Dict

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient

from forge_env.models import ForgeAction, ForgeObservation, ForgeState


class ForgeEnvClient(EnvClient[ForgeAction, ForgeObservation, ForgeState]):
    """WebSocket client for a remote Forge OpenEnv server (HF Space or local Docker)."""

    def _step_payload(self, action: ForgeAction) -> Dict[str, Any]:
        return action.model_dump(mode="json")

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[ForgeObservation]:
        obs_raw = payload.get("observation", {})
        return StepResult(
            observation=ForgeObservation.model_validate(obs_raw),
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> ForgeState:
        return ForgeState.model_validate(payload)


# Backward-compatible alias
ForgeEnv = ForgeEnvClient
