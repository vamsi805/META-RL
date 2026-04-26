from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from openenv.core.env_server.types import Action, Observation, State
from pydantic import ConfigDict, Field


class ForgeAction(Action):
    model_config = ConfigDict(extra="ignore")

    type: Literal["execute", "propose_edit", "validate", "commit", "rollback"]
    edit: Optional[Dict[str, Any]] = None
    task_id: Optional[str] = None


class ForgeObservation(Observation):
    model_config = ConfigDict(extra="ignore")

    phase: str
    playbook_snapshot: Dict[str, Any] = Field(default_factory=dict)
    last_trajectory: Optional[Dict[str, Any]] = None
    last_score: Optional[float] = None
    val_baseline: float = 0.0
    val_after_edit: Optional[float] = None
    reward_components: Optional[Dict[str, float]] = None
    failures_buffer: List[Dict[str, Any]] = Field(default_factory=list)


class ForgeState(State):
    model_config = ConfigDict(extra="ignore")

    playbook_version: int = 0
    edit_log: List[Dict[str, Any]] = Field(default_factory=list)
    rolling_val_accuracy: float = 0.0
    rolling_complexity: int = 0
    curriculum_step: int = 0
    curriculum_phase: str = ""
