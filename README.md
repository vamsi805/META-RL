# Forge

Forge is an **OpenEnv** environment where an **Editor** proposes structured edits to a versioned **Playbook** (prompts, lessons, tools). Each candidate edit is scored by running an **Executor** on a fixed validation slice and optional hold-out tasks, with a **three-term reward** (validation gain, complexity penalty, regression penalty) plus a small diversity bonus.

This repo implements the hackathon **MVP through P2**: curriculum, multi-tier tasks (including hold-out template families), ensemble T2 grading, a bounded Executor loop, GRPO-style **group scoring** over candidate edits, sandboxed tool authoring, plotting, Gradio demo, and CI. **P3** items (open-ended tool authoring polish, full TRL `GRPOTrainer` fine-tune, MCP) are intentionally out of scope here.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
export FORGE_TOOL_ISOLATION=subprocess   # or `auto` if Docker is available
pytest -q
python training/train_forge.py --smoke
python outputs/plots/generate_plots.py
python demo/app.py
```

## Real model mode

Default mode is `stub`, which avoids downloading a model. To use Qwen through local Transformers:

```bash
pip install -e ".[train,dev]"
export FORGE_LLM_BACKEND=transformers
export FORGE_LLM_MODEL=Qwen/Qwen2.5-1.5B-Instruct
export FORGE_TOOL_ISOLATION=subprocess
python training/train_forge.py --llm_backend transformers --model_name "$FORGE_LLM_MODEL" --max_steps 3
```

For Colab T4, Hugging Face credits, and push steps, see **[docs/REAL_LLM_AND_COLAB.md](docs/REAL_LLM_AND_COLAB.md)**.

## Architecture overview

| Piece | Role |
|--------|------|
| `forge_env/server/forge_environment.py` | OpenEnv `Environment`: `reset` / `step` / `state`, curriculum-aware eval, `preview_propose` for GRPO groups |
| `forge_env/server/executor.py` | Bounded multi-step rollout; real model mode can call tools and feed tool results back to the model |
| `forge_env/server/edit_ops.py` | Apply structured edits; `add_tool` runs AST + smoke test + optional Docker isolation |
| `forge_env/server/tasks/` | T1 programmatic, T2 rubric (ensemble heuristics), T3 hidden-goals; **alpha** vs **beta** template families |
| `forge_env/server/curriculum.py` | Warm-up (T1) → diversification (+T2) → full (+T3) by global step |
| `forge_env/server/sandbox/` | Tool execution: Docker (`network=none`, `read-only`, default seccomp) or subprocess fallback |
| `training/train_forge.py` | GRPO-style loop: sample K edits, `preview_propose` each, group-relative advantages, commit best if positive |
| `outputs/plots/generate_plots.py` | PRD-style figures from `outputs/smoke_metrics.jsonl` |
| `demo/app.py` | Gradio: evolution summary, manifest viewer, prompt diff, metrics replay |

Full **call-level flow** (what file calls what, and when): see **[docs/ARCHITECTURE_AND_FLOW.md](docs/ARCHITECTURE_AND_FLOW.md)**.

## OpenEnv server

```bash
uvicorn forge_env.server.app:app --host 0.0.0.0 --port 8000
```

The app is built with `openenv.core.env_server.create_fastapi_app`. Remote clients: `ForgeEnvClient` in `forge_env/client.py`.

## Docker

**Server** (isolates the env process from the host):

```bash
docker build -f forge_env/server/Dockerfile -t forge-env .
docker run --rm -p 8000:8000 --read-only --tmpfs /tmp:rw,size=64m forge-env
```

**Tool runs** (when `FORGE_TOOL_ISOLATION=auto` or `docker`): see README section in earlier commits — tools run in a throwaway container with `--network none`, read-only root, tmpfs `/tmp`, and Docker’s default seccomp profile.

## Submission checklist (hackathon-oriented)

- [x] OpenEnv base classes + `create_fastapi_app`
- [x] Training script + metrics JSONL + plots
- [ ] Host on Hugging Face Space (`openenv push`) — you still need to push this repo’s package
- [ ] W&B / video / blog links — add URLs here when available

## License

BSD-style (match OpenEnv / team choice).
