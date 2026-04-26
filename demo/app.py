import gradio as gr

from demo.components.data_loader import load_metrics
from demo.components.playbook_browser import (
    list_playbook_versions,
    load_manifest,
    metrics_tail_jsonl,
    prompt_diff_v0_vs_latest,
)


def evolution_tree_text():
    versions = list_playbook_versions()
    if not versions:
        return "No checkpoints under outputs/playbook_versions/. Run training first."
    lines = ["Playbook evolution (newest last):"]
    for v in versions:
        lines.append(f"  • {v}")
    rows = load_metrics()
    if rows:
        lines.append(f"\nLast training step: {rows[-1].get('step')}, reward={rows[-1].get('reward')}")
    return "\n".join(lines)


def manifest_view(version: str):
    v = (version or "").strip()
    if not v:
        return "Enter a version folder name (e.g. v3)."
    return load_manifest(v) or "(empty)"


def replay_text():
    rows = load_metrics()
    if not rows:
        return "No outputs/smoke_metrics.jsonl — run training/train_forge.py"
    lines = []
    for r in rows[-12:]:
        acc = r.get("val_after_edit") or r.get("val_baseline")
        ho = r.get("holdout_accuracy")
        ho_s = f"{ho:.3f}" if isinstance(ho, (int, float)) else str(ho)
        lines.append(
            f"step={r['step']} phase={r.get('curriculum_phase', '?')} "
            f"val={float(acc):.3f} holdout={ho_s} "
            f"reward={float(r['reward']):.3f} accepted={r.get('edit_accepted')}"
        )
    return "\n".join(lines)


def run_try(task: str):
    return (
        "Demo stub: wire an LLM executor here (see docs/ARCHITECTURE_AND_FLOW.md). "
        f"Task was: {task[:200]}"
    )


with gr.Blocks(title="Forge Demo") as app:
    gr.Markdown("# Forge — playbook evolution demo")
    with gr.Tab("Playbook evolution tree"):
        tree_out = gr.Textbox(label="Tree summary", lines=10)
        tree_btn = gr.Button("Refresh tree")
        tree_btn.click(fn=evolution_tree_text, outputs=tree_out)
        gr.Markdown("### Checkpoint manifest")
        ver_in = gr.Textbox(label="Version folder (e.g. v3)", placeholder="v3")
        man_out = gr.Textbox(label="manifest.yaml", lines=16)
        man_btn = gr.Button("Load manifest")
        man_btn.click(fn=manifest_view, inputs=ver_in, outputs=man_out)
    with gr.Tab("Prompt diff gallery"):
        diff_btn = gr.Button("Diff reflector: v0 disk vs latest checkpoint")
        diff_out = gr.Code(label="Unified diff", language="diff")
        diff_btn.click(fn=prompt_diff_v0_vs_latest, outputs=diff_out)
    with gr.Tab("Trajectory / metrics replay"):
        rep = gr.Textbox(label="Recent metrics rows", lines=14)
        rep_btn = gr.Button("Load training metrics summary")
        rep_btn.click(fn=replay_text, outputs=rep)
        raw_out = gr.Code(label="Raw JSONL tail", language="json")
        raw_btn = gr.Button("Raw JSONL tail")
        raw_btn.click(fn=lambda: metrics_tail_jsonl(20), outputs=raw_out)
    with gr.Tab("Live training dashboard"):
        gr.Markdown(
            "Embed your Weights & Biases project here (e.g. an iframe src to your run URL). "
            "Plots are saved under `outputs/plots/` by `outputs/plots/generate_plots.py`."
        )
    with gr.Tab("Try it yourself"):
        inp = gr.Textbox(label="Task", lines=2)
        out = gr.Textbox(label="Agent output", lines=4)
        run_btn = gr.Button("Run (stub)")
        run_btn.click(fn=run_try, inputs=inp, outputs=out)


if __name__ == "__main__":
    app.launch()
