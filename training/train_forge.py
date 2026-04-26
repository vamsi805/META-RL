from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from forge_env.models import ForgeAction, ForgeObservation
from forge_env.server.curriculum import phase_name
from forge_env.server.forge_environment import ForgeEnvironment
from training.editor_policy import llm_candidate_edits, stub_candidate_edits
from training.grpo_config import ForgeTrainConfig
from training.unsloth_setup import maybe_patch_unsloth
from training.trl_bridge import describe_trl_status


def policy_candidate_edits(step: int, k: int, env: ForgeEnvironment, cfg: ForgeTrainConfig):
    """Return model-written edits when enabled, otherwise deterministic local edits."""
    if cfg.llm_backend in {"", "stub", "none"}:
        return stub_candidate_edits(step, k)
    return llm_candidate_edits(
        step=step,
        k=k,
        playbook=env.playbook,
        failures_buffer=env.failures_buffer,
        val_baseline=env.val_baseline,
        backend=cfg.llm_backend,
        model_name=cfg.model_name,
    )


def run_training(cfg: ForgeTrainConfig, smoke: bool = False) -> Path:
    os.environ["FORGE_LLM_BACKEND"] = cfg.llm_backend
    os.environ["FORGE_LLM_MODEL"] = cfg.model_name
    env = ForgeEnvironment()
    enabled, _ = maybe_patch_unsloth()
    describe_trl_status(cfg.model_name)
    total_steps = cfg.smoke_steps if smoke else min(cfg.max_steps, 400)
    k = 4 if smoke else min(cfg.num_generations, 8)

    logs: List[dict] = []
    env.reset(training_step=0)

    for step in range(total_steps):
        env.set_curriculum_step(step)
        candidates = policy_candidate_edits(step, k, env, cfg)
        previews = [env.preview_propose(c) for c in candidates]
        rewards = [p[0] for p in previews]
        mean_r = sum(rewards) / max(len(rewards), 1)
        advantages = [r - mean_r for r in rewards]
        best_i = max(range(len(rewards)), key=lambda i: rewards[i])
        best_reward = rewards[best_i]
        best_advantage = advantages[best_i] if advantages else 0.0
        # This is a selection-loss proxy for the playbook-edit RL loop. It is not a
        # neural-network training loss unless a future trainer updates model weights.
        loss_proxy = float(max(0.0, -best_advantage))

        if best_reward > 0:
            obs = env.step(
                ForgeAction(type="propose_edit", edit=candidates[best_i]),
                training_step=step,
            )
        else:
            obs = ForgeObservation(
                phase="no_commit",
                playbook_snapshot=env.playbook.snapshot(),
                failures_buffer=list(env.failures_buffer),
                val_baseline=env.val_baseline,
                val_after_edit=env.val_baseline,
                reward=0.0,
                done=False,
                metadata={"curriculum_phase": phase_name(step)},
            )

        val_after = obs.val_after_edit
        if val_after is None:
            val_after = obs.val_baseline

        row = {
            "step": step,
            "reward": float(obs.reward),
            "val_baseline": obs.val_baseline,
            "val_after_edit": val_after,
            "holdout_accuracy": env._evaluate(env.playbook, env.splits["holdout"]),
            "val_by_tier": env.tier_accuracy(env.playbook, env.splits["val"]),
            "playbook_token_count": env.playbook.token_count(),
            "edit_accepted": int(best_reward > 0),
            "grpo_group_mean": mean_r,
            "grpo_advantages": advantages,
            "grpo_best_index": best_i,
            "selection_loss_proxy": loss_proxy,
            "curriculum_phase": phase_name(step),
            "components": obs.reward_components or {},
            "playbook_version": env.state.playbook_version,
            "unsloth_enabled": enabled,
        }
        logs.append(row)
        if obs.done:
            break

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    log_path = out_dir / "smoke_metrics.jsonl"
    with log_path.open("w", encoding="utf-8") as f:
        for row in logs:
            f.write(json.dumps(row) + "\n")
    return log_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument(
        "--llm_backend",
        choices=["stub", "transformers", "hf_api"],
        default=os.environ.get("FORGE_LLM_BACKEND", "stub"),
    )
    parser.add_argument("--max_steps", type=int, default=400)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = ForgeTrainConfig(
        model_name=args.model_name,
        max_steps=args.max_steps,
        llm_backend=args.llm_backend,
    )
    path = run_training(cfg, smoke=args.smoke)
    print(f"Wrote metrics to {path}")
