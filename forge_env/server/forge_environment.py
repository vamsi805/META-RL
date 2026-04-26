from __future__ import annotations

import random
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openenv.core.env_server import Environment

from forge_env.models import ForgeAction, ForgeObservation, ForgeState
from forge_env.server.budgets import violates_budget
from forge_env.server.config import MAX_EPISODE_STEPS
from forge_env.server.curriculum import active_tiers_for_step, phase_name
from forge_env.server.edit_ops import ToolValidationError, apply_edit
from forge_env.server.executor import Executor
from forge_env.server.persistence import write_playbook_checkpoint
from forge_env.server.playbook import Playbook, load_initial_playbook
from forge_env.server.rewards.compose import compute_reward
from forge_env.server.tasks.splits import build_task_splits
from forge_env.server.tasks.tier1_programmatic import grade_tier1
from forge_env.server.tasks.tier2_rubric import grade_tier2_ensemble
from forge_env.server.tasks.tier3_simuser import grade_tier3


class ForgeEnvironment(Environment[ForgeAction, ForgeObservation, ForgeState]):
    """OpenEnv Environment: playbook edit loop with validation reward."""

    SUPPORTS_CONCURRENT_SESSIONS = False

    def __init__(self) -> None:
        super().__init__()
        package_root = Path(__file__).resolve().parents[1]
        self._package_root = package_root
        self.playbook: Playbook = load_initial_playbook(package_root)
        self.splits = build_task_splits(base_seed=42)
        self.failures_buffer: List[Dict[str, Any]] = []
        self._step_count = 0
        self._rng = random.Random(42)
        self.executor = Executor()
        self._curriculum_step = 0
        self.val_baseline = self._evaluate(self.playbook, self.splits["val"])
        self._episode_id: Optional[str] = None
        self._regression_tasks: List[Dict[str, Any]] = []

    def set_curriculum_step(self, step: int) -> None:
        self._curriculum_step = int(step)

    def _active_tiers(self) -> set[str]:
        return active_tiers_for_step(self._curriculum_step)

    def _filter_tasks(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        tiers = self._active_tiers()
        return [t for t in tasks if t.get("tier") in tiers]

    def _predict(self, task: Dict[str, Any], playbook: Playbook) -> Dict[str, Any]:
        pred, _traj = self.executor.run(task, playbook)
        return pred

    def _grade(self, task: Dict[str, Any], prediction: Dict[str, Any]) -> float:
        if task["tier"] == "T1":
            return grade_tier1(prediction, task["expected"])
        if task["tier"] == "T2":
            return grade_tier2_ensemble(prediction.get("text", ""), task["rubric"])
        return grade_tier3(prediction.get("text", ""), task)

    def _evaluate(self, playbook: Playbook, tasks: List[Dict[str, Any]]) -> float:
        filtered = self._filter_tasks(tasks)
        if not filtered:
            return 0.0
        scores = []
        for task in filtered:
            pred = self._predict(task, playbook)
            scores.append(self._grade(task, pred))
        return float(sum(scores) / len(scores))

    def tier_accuracy(
        self,
        playbook: Playbook,
        tasks: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        filtered = self._filter_tasks(tasks)
        out: Dict[str, float] = {}
        for tier in ("T1", "T2", "T3"):
            sub = [t for t in filtered if t.get("tier") == tier]
            if sub:
                out[tier] = float(
                    sum(self._grade(t, self._predict(t, playbook)) for t in sub) / len(sub)
                )
        return out

    def _failure_count_on_tasks(self, playbook: Playbook, tasks: List[Dict[str, Any]]) -> int:
        n = 0
        for task in tasks:
            pred = self._predict(task, playbook)
            if self._grade(task, pred) < 0.7:
                n += 1
        return n

    def _collect_failures(self, n: int = 4) -> List[Dict[str, Any]]:
        train_f = self._filter_tasks(self.splits["train"])
        if not train_f:
            return []
        sampled = self._rng.sample(train_f, k=min(n, len(train_f)))
        failures: List[Dict[str, Any]] = []
        for task in sampled:
            pred = self._predict(task, self.playbook)
            score = self._grade(task, pred)
            if score < 0.7:
                failures.append({"task": task, "prediction": pred, "score": score})
        return failures

    def _diversity_distance(self, edit: Dict[str, Any]) -> float:
        recent = [e.get("action") for e in self.playbook.edit_log[-5:]]
        atype = edit.get("type", "no_op")
        if not recent or atype not in recent:
            return 1.0
        return 0.25

    def _regression_delta(self, old_pb: Playbook, candidate_pb: Playbook) -> int:
        old_fail = self._failure_count_on_tasks(old_pb, self._regression_tasks)
        new_fail = self._failure_count_on_tasks(candidate_pb, self._regression_tasks)
        return max(0, new_fail - old_fail)

    def _score_candidate(
        self,
        old_pb: Playbook,
        candidate_pb: Playbook,
        edit: Dict[str, Any],
    ) -> Tuple[float, Dict[str, float], float]:
        val_tasks = self._filter_tasks(self.splits["val"])
        new_val = self._evaluate(candidate_pb, val_tasks)
        reg = self._regression_delta(old_pb, candidate_pb)
        reward, components = compute_reward(
            old_pb=old_pb,
            new_pb=candidate_pb,
            baseline_val=self.val_baseline,
            new_val=new_val,
            newly_failing_tasks=reg,
            diversity_distance=self._diversity_distance(edit),
        )
        return reward, components, new_val

    def preview_propose(self, edit: Dict[str, Any]) -> Tuple[float, Dict[str, float], float]:
        """Score an edit without mutating the committed playbook (for GRPO group scoring)."""
        old = self.playbook
        try:
            candidate = apply_edit(old, edit)
        except ToolValidationError:
            return -0.2, {"total": -0.2}, self.val_baseline
        if violates_budget(candidate):
            return -1.0, {"total": -1.0}, self.val_baseline
        return self._score_candidate(old, candidate, edit)

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> ForgeObservation:
        if seed is not None:
            self._rng = random.Random(seed)
        self._episode_id = episode_id or str(uuid.uuid4())
        self._step_count = 0
        self._curriculum_step = int(kwargs.get("training_step", 0))
        self.playbook = load_initial_playbook(self._package_root)
        train_f = self._filter_tasks(self.splits["train"])
        train_sorted = sorted(train_f, key=lambda t: t["task_id"])
        self._regression_tasks = train_sorted[: min(24, len(train_sorted))]
        self.failures_buffer = self._collect_failures(n=6)
        self.val_baseline = self._evaluate(self.playbook, self.splits["val"])
        return ForgeObservation(
            phase="awaiting_edit",
            playbook_snapshot=self.playbook.snapshot(),
            failures_buffer=self.failures_buffer,
            val_baseline=self.val_baseline,
            done=False,
            reward=0.0,
            metadata={
                "curriculum_phase": phase_name(self._curriculum_step),
                "active_tiers": sorted(self._active_tiers()),
            },
        )

    def step(
        self,
        action: ForgeAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> ForgeObservation:
        if "training_step" in kwargs:
            self.set_curriculum_step(int(kwargs["training_step"]))

        self._step_count += 1
        if action.type != "propose_edit" or action.edit is None:
            return ForgeObservation(
                phase="idle",
                playbook_snapshot=self.playbook.snapshot(),
                failures_buffer=self.failures_buffer,
                val_baseline=self.val_baseline,
                done=(self._step_count >= MAX_EPISODE_STEPS),
                reward=0.0,
                metadata={"curriculum_phase": phase_name(self._curriculum_step)},
            )

        old_playbook = self.playbook
        try:
            candidate = apply_edit(old_playbook, action.edit)
        except ToolValidationError as exc:
            return ForgeObservation(
                phase="tool_rejected",
                playbook_snapshot=self.playbook.snapshot(),
                failures_buffer=self.failures_buffer,
                val_baseline=self.val_baseline,
                reward=-0.2,
                reward_components={"total": -0.2},
                metadata={
                    "tool_validation_error": str(exc),
                    "curriculum_phase": phase_name(self._curriculum_step),
                },
                done=(self._step_count >= MAX_EPISODE_STEPS),
            )

        if violates_budget(candidate):
            return ForgeObservation(
                phase="budget_violation",
                playbook_snapshot=self.playbook.snapshot(),
                failures_buffer=self.failures_buffer,
                val_baseline=self.val_baseline,
                reward=-1.0,
                reward_components={"total": -1.0},
                done=(self._step_count >= MAX_EPISODE_STEPS),
                metadata={"curriculum_phase": phase_name(self._curriculum_step)},
            )

        reward, components, new_val = self._score_candidate(
            old_playbook, candidate, action.edit
        )
        if reward > 0:
            self.playbook = candidate
            self.val_baseline = new_val
            self.failures_buffer = self._collect_failures(n=10)
            write_playbook_checkpoint(self.playbook, components)

        return ForgeObservation(
            phase="edit_evaluated",
            playbook_snapshot=self.playbook.snapshot(),
            failures_buffer=self.failures_buffer,
            val_baseline=self.val_baseline,
            val_after_edit=new_val,
            reward_components=components,
            reward=reward,
            done=(self._step_count >= MAX_EPISODE_STEPS),
            metadata={
                "curriculum_phase": phase_name(self._curriculum_step),
                "edit_accepted": reward > 0,
            },
        )

    @property
    def state(self) -> ForgeState:
        return ForgeState(
            episode_id=self._episode_id,
            step_count=self._step_count,
            playbook_version=self.playbook.version,
            edit_log=self.playbook.edit_log,
            rolling_val_accuracy=self.val_baseline,
            rolling_complexity=self.playbook.token_count(),
            curriculum_step=self._curriculum_step,
            curriculum_phase=phase_name(self._curriculum_step),
        )
