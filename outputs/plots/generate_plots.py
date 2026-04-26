from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib").resolve()))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def generate(metrics_path: Path | None = None) -> None:
    path = metrics_path or Path("outputs/smoke_metrics.jsonl")
    metrics = _load(path)
    out = Path("outputs/plots")
    out.mkdir(parents=True, exist_ok=True)
    if not metrics:
        return

    steps = [m["step"] for m in metrics]
    val = [m.get("val_after_edit") if m.get("val_after_edit") is not None else m["val_baseline"] for m in metrics]
    hold = [m.get("holdout_accuracy", 0.0) for m in metrics]
    baseline_val = metrics[0].get("val_baseline", val[0])

    # 01 Validation + holdout
    plt.figure(figsize=(7, 4))
    plt.plot(steps, val, label="validation")
    plt.plot(steps, hold, label="holdout (unseen templates)")
    plt.axhline(baseline_val, color="gray", linestyle="--", label="initial val baseline")
    plt.xlabel("Training step")
    plt.ylabel("Accuracy")
    plt.title("Validation vs holdout")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "01_val_accuracy.png")
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(steps, hold, label="holdout", color="darkorange")
    plt.xlabel("Training step")
    plt.ylabel("Holdout accuracy")
    plt.title("Holdout generalisation (beta templates)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "02_holdout_generalisation.png")
    plt.close()

    # 03 Stacked reward components (absolute contributions from last row scale — use per-step bars cumulatively as line plot)
    dval = [m.get("components", {}).get("delta_val", 0.0) for m in metrics]
    comp = [m.get("components", {}).get("complexity", 0.0) for m in metrics]
    reg = [m.get("components", {}).get("regression", 0.0) for m in metrics]
    div = [m.get("components", {}).get("diversity", 0.0) for m in metrics]
    plt.figure(figsize=(7, 4))
    plt.plot(steps, dval, label="Δval (term)")
    plt.plot(steps, comp, label="complexity (term)")
    plt.plot(steps, reg, label="regression (term)")
    plt.plot(steps, div, label="diversity (term)")
    plt.xlabel("Training step")
    plt.ylabel("Reward component value")
    plt.title("Reward components over time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "03_reward_components_stacked.png")
    plt.close()

    tok = [m.get("playbook_token_count", 0) for m in metrics]
    plt.figure(figsize=(7, 4))
    plt.plot(steps, tok, label="playbook tokens (est.)")
    plt.xlabel("Training step")
    plt.ylabel("Estimated tokens")
    plt.title("Playbook complexity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "04_playbook_complexity.png")
    plt.close()

    # 05 Per-tier breakdown (last step snapshot)
    last = metrics[-1]
    tiers = last.get("val_by_tier") or {}
    if tiers:
        plt.figure(figsize=(6, 4))
        names = list(tiers.keys())
        vals = [tiers[k] for k in names]
        plt.bar(names, vals, color=["#4C72B0", "#55A868", "#C44E52"])
        plt.ylabel("Val accuracy (subset)")
        plt.title("Per-tier validation (final step)")
        plt.tight_layout()
        plt.savefig(out / "05_per_tier_breakdown.png")
        plt.close()

    accept = [100.0 * m.get("edit_accepted", 0) for m in metrics]
    plt.figure(figsize=(7, 4))
    plt.plot(steps, accept, label="edit accepted %")
    plt.xlabel("Training step")
    plt.ylabel("Accepted (%)")
    plt.title("Edit acceptance rate")
    plt.tight_layout()
    plt.savefig(out / "06_edit_acceptance.png")
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(steps, val, label="trained trajectory")
    plt.axhline(baseline_val, color="gray", linestyle="--", label="run start baseline")
    plt.xlabel("Training step")
    plt.ylabel("Validation accuracy")
    plt.title("Baseline vs trained (same axes)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "07_baseline_vs_trained.png")
    plt.close()

    loss = [m.get("selection_loss_proxy", 0.0) for m in metrics]
    plt.figure(figsize=(7, 4))
    plt.plot(steps, loss, label="selection loss proxy", color="crimson")
    plt.xlabel("Training step")
    plt.ylabel("Loss proxy")
    plt.title("Training loss proxy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "08_training_loss_proxy.png")
    plt.close()


if __name__ == "__main__":
    generate()
