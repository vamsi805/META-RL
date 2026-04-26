# Forge — A Self-Improving Agent Harness

## Product Requirements Document (v1.0)

**Submission target:** Meta × Hugging Face OpenEnv Hackathon (India, Apr 2026)
**Theme alignment:** Theme 4 (Self-Improvement) primary, Theme 3.2 (Personalized Tasks) secondary
**Document owner:** [Team Forge]
**Last updated:** April 2026

---

## 0. TL;DR

Forge is an OpenEnv environment in which an LLM agent improves at personal-assistant tasks not by updating its weights, but by **editing its own scaffolding** — system prompts, tool library, lesson notes, and routing rules — based on a feedback signal grounded in a held-out validation set.

The agent operates a versioned artifact called a **Playbook**. After every batch of task attempts, an **Editor policy** proposes structured edits to the Playbook (`edit_prompt`, `add_tool`, `write_lesson`, etc.). Each candidate edit is validated against a held-out task slice and rewarded by **Δ validation accuracy − complexity penalty − regression penalty**. Editor parameters are updated with GRPO (TRL) on top of Unsloth-optimised Qwen2.5-3B.

The deliverable is:
1. An OpenEnv-compliant environment (deployed as a Hugging Face Space)
2. A working TRL/Unsloth training Colab that produces a measurable reward curve
3. A live Gradio demo that visualises the **playbook evolution tree** — the headline storytelling artefact
4. A README, blog post, and 2-minute video

---

## 1. Problem & Motivation

### 1.1 The capability gap

Current agentic LLMs are deployed inside hand-tuned harnesses (OpenHands, OpenClaw, ReAct loops). Every failure mode requires a human engineer to rewrite the system prompt, add a tool, or patch a memory policy. The agent itself never learns to edit its own scaffolding. This is a hard ceiling on agent autonomy.

### 1.2 Why this is the right problem for the hackathon

| Hackathon criterion | How Forge scores |
|---|---|
| **Innovation (40%)** | No environment on the OpenEnv Hub trains self-modifying scaffolding. Closest prior work (Voyager, Absolute Zero Reasoner, Self-Challenging LM Agents) is paper-only and not OpenEnv-compatible. |
| **Storytelling (30%)** | The Playbook is human-readable. We can literally show "the prompt at step 0" vs "the prompt at step 500" and the new tools the agent invented. |
| **Reward improvement (20%)** | Reward is a clean Δ on a held-out task set — easy to plot, hard to game. |
| **Pipeline coherence (10%)** | Three independent reward functions, sandboxed tool execution, programmatic verification. Maps cleanly onto the help-guide's anti-reward-hacking checklist. |

### 1.3 Why this is feasible

The action space (structured edits to a small set of files) is *much* smaller than "generate arbitrary Python from scratch." The reward is *fully verifiable* (re-run the agent on a held-out set, compute accuracy). The base model can be Qwen2.5-3B-Instruct at 4-bit on a single Colab T4. **No data labelling required.**

---

## 2. Concept

### 2.1 The dual-artifact frame

Forge decomposes an agent into two artefacts:

- **The Engine** — LLM weights + an inference loop. Frozen during a Forge episode (or trained on a slow outer loop, see §11.5).
- **The Playbook** — a versioned, modular, file-based bundle of prompts, tools, and lessons. This is the *learned object*.

A Playbook is data, not code-of-the-model. It is diff-able, roll-back-able, branchable, and human-readable. It compresses "what this agent has learned about how to behave."

### 2.2 The two roles

The same model weights are used in two modes via different prompts:

- **Executor** — given (Task, Playbook), runs a multi-turn rollout and produces a Trajectory.
- **Editor** — given (failed Trajectory, current Playbook), proposes a structured edit.

We train the Editor with GRPO. The Executor is fixed (or co-trained on a slower loop).

### 2.3 The edit-and-validate loop in one paragraph

Run K Executor rollouts on a batch of training tasks → keep the failures → ask the Editor for N candidate edits per failure → for each candidate edit, instantiate a temporary Playbook and run the Executor on a small held-out validation slice → compute Δ-validation reward → use GRPO group-relative advantages to update the Editor → commit the highest-reward edit if it's positive.

---

## 3. The Playbook (Editable Artefact)

### 3.1 Directory structure

```
playbook/
├── manifest.yaml           # version, parent_hash, edit_log[]
├── prompts/
│   ├── planner.md          # "decompose the task into sub-goals"
│   ├── executor.md         # "take one action at a time"
│   ├── reflector.md        # "after each step, critique"
│   ├── formatter.md        # "format your final answer as ..."
│   └── persona.md          # tone, voice, principles
├── tools/
│   ├── _registry.yaml      # which tools are active + descriptions
│   ├── builtin/            # FROZEN — never edited
│   │   ├── send_email.py
│   │   ├── search_calendar.py
│   │   ├── lookup_contact.py
│   │   └── ...
│   └── learned/            # agent-authored
│       └── (initially empty)
├── lessons/
│   ├── general.md          # cross-task heuristics
│   └── per_domain/
│       ├── scheduling.md
│       ├── email_replies.md
│       └── shopping.md
└── routing.yaml            # task_type → which prompts/tools to load
```

### 3.2 Manifest schema

```yaml
# manifest.yaml
version: 47
parent: 46
created_at: 2026-04-25T14:23:11Z
edit:
  type: edit_prompt
  target: prompts/reflector.md
  rationale: "Executor was missing implicit deadlines in emails"
  reward_components:
    delta_val: +0.083
    complexity: -0.005
    regression: 0.0
    total: +0.078
metrics_snapshot:
  val_accuracy: 0.612
  prompt_token_count: 1843
  tool_count: 12
  lesson_count: 6
```

### 3.3 Initial Playbook (v0)

The starting Playbook is deliberately minimal: a 3-section system prompt (planner, executor, reflector), a fixed registry of ~15 builtin tools, and one empty `lessons/general.md`. We want the agent to discover useful additions, not to start from a prebuilt expert configuration.

### 3.4 Complexity budget

Hard caps to prevent runaway bloat:
- Total prompt tokens ≤ 4,000
- Tool count (builtin + learned) ≤ 30
- Per-tool LOC ≤ 80
- Lesson tokens ≤ 2,000

Edits that violate the budget are auto-rejected.

---

## 4. The Engine

### 4.1 Executor inference loop

```
loop:
  prompt = render(playbook.prompts, task)
  available_tools = playbook.tools.active()
  action = LLM(prompt, history, available_tools)
  if action.is_tool_call:
    obs = sandboxed_execute(action.tool, action.args)
    history.append(action, obs)
  else:
    return action.final_answer
  if step_count > MAX_STEPS: break
```

Max steps per episode: 12 (sufficient for our task tier, capped to keep rollouts cheap).

### 4.2 Editor prompt template

The Editor receives:
- Current Playbook (rendered with line numbers)
- Up to 3 failed trajectories with task descriptions and rubric scores
- A diagnostic prompt asking it to (a) identify the root cause and (b) emit ONE structured edit action

Output format (JSON, schema-validated):

```json
{
  "diagnosis": "The reflector never re-reads the original task before finalising, so it misses constraints introduced near the start.",
  "action": {
    "type": "edit_prompt",
    "target": "prompts/reflector.md",
    "new_content": "...",
    "expected_improvement": "tasks where the user states constraints upfront"
  }
}
```

If parsing fails, the action defaults to `no_op` and gets reward 0.

---

## 5. Task Distribution

### 5.1 Three tiers

Following the help-guide's *"prefer crisp verification"* rule, we weight toward objective grading:

| Tier | % of tasks | Verification | Examples |
|---|---|---|---|
| **T1: Programmatic** | 30% | Hard equality / regex / code execution | Schedule meeting given constraints; extract structured data from email; convert currencies |
| **T2: Rubric-judged** | 50% | Ensemble of 3 LLM judges with structured rubric | Reply to passive-aggressive email; draft polite refusal; summarise long thread |
| **T3: Simulated-user** | 20% | Multi-turn vs hidden-persona LLM; goal-completion check | Negotiate refund with simulated CS rep; gather requirements from picky user |

### 5.2 Procedural generation

Each tier uses **templates with randomised slots**:

```
T1.scheduling template:
  participants: sample(3-5 names from pool)
  windows: random(weekday × hour-blocks, ensure ≥1 valid intersection)
  duration: choice(30, 45, 60, 90 min)
  constraints: choice(no-after-5pm, no-mondays, only-mornings, none)
```

This gives ~10⁴ unique instances per template. Training set, validation set, and held-out test set use the same templates with **disjoint random seeds**.

### 5.3 Splits

- **Train pool** — 200 task instances, refreshed every 50 training steps
- **Validation slice** — 40 instances, FIXED across training (this is what reward is computed against)
- **Held-out generalisation set** — 60 instances drawn from templates not seen during validation (for final demo plot)

### 5.4 Why this avoids reward hacking

- Validation set is **separate** from training rollouts — agent can't memorise
- Held-out set has **different templates entirely** — tests real generalisation
- T1 tasks have **deterministic ground truth** — judge-bias-free
- Multiple judges in T2 + temperature variance — single-judge gaming is harder
- Edits are scored against tasks the Executor already failed — only generalising patches survive

---

## 6. The Edit-and-Validate Loop

### 6.1 Algorithm (one Forge episode = one edit-and-validate cycle)

```
def forge_episode(playbook_t):
    # PHASE A — collect failures
    failures = []
    for task in sample(train_pool, 8):
        traj = executor.run(task, playbook_t)
        score = grade(traj, task)
        if score < 0.7:
            failures.append((task, traj, score))
    if not failures:
        return playbook_t, reward=0  # already strong, nothing to learn

    # PHASE B — propose K candidate edits
    candidates = [
        editor.propose(failures, playbook_t)
        for _ in range(K=8)
    ]

    # PHASE C — validate each candidate
    baseline_val = eval(executor, playbook_t, val_set)
    for c in candidates:
        c.playbook = apply_edit(playbook_t, c.action)
        if violates_budget(c.playbook):
            c.reward = -1.0
            continue
        new_val = eval(executor, c.playbook, val_set)
        c.delta = new_val - baseline_val
        c.complexity_pen = lambda1 * delta_tokens(c.playbook, playbook_t)
        c.regression_pen = lambda2 * count_newly_failing(c.playbook, playbook_t)
        c.reward = c.delta - c.complexity_pen - c.regression_pen

    # PHASE D — GRPO update on Editor
    advantages = group_relative(candidates.rewards)
    grpo_step(editor, candidates, advantages)

    # PHASE E — commit best edit if positive
    best = max(candidates, key=lambda c: c.reward)
    return (best.playbook if best.reward > 0 else playbook_t), best.reward
```

### 6.2 The action space (Editor outputs)

Six action types:

| Type | Targets | Effect |
|---|---|---|
| `edit_prompt` | any file in `prompts/` | replaces the file content |
| `write_lesson` | `lessons/general.md` or `lessons/per_domain/X.md` | appends a paragraph |
| `add_tool` | `tools/learned/` | creates new Python tool with smoke test |
| `modify_tool` | `tools/learned/X.py` only (builtin frozen) | applies diff |
| `edit_routing` | `routing.yaml` | changes which prompts/tools load per task type |
| `no_op` | — | makes no change (allowed; sometimes the right answer) |

### 6.3 Reward function (the heart)

Per-candidate reward:

```
R(c) = α · Δval(c)
     − β · complexity_penalty(c)
     − γ · regression_penalty(c)
     + δ · diversity_bonus(c)        # only at edit-attribution time
```

With:
- `α = 1.0` — main signal
- `β = 0.05 · (new_token_count / 100)` — soft, scales with bloat
- `γ = 0.5 · (newly_failing_tasks)` — asymmetric, regressions hurt 5× more than gains
- `δ = 0.02 × dist(c, recent_committed_edits)` — encourages exploration

These three penalties are the **multiple-independent-reward-functions** the help guide demands. They each defend a different failure mode (bloat, regression, mode collapse).

### 6.4 Reward-hacking defences (mapped from help guide §8)

| Hack the agent might attempt | Our defence |
|---|---|
| Memorise validation tasks | Validation set is held out from training rollouts; held-out set uses unseen templates |
| Add bloat to make tools "look better" | Complexity penalty + hard token cap |
| Break old tasks to "fix" new ones | Asymmetric regression penalty |
| Author tool that calls `os.system` to mutate state | Sandboxed subprocess, no FS/network for learned tools, AST static analysis |
| Always emit `no_op` if rewards are noisy | δ-bonus rewards exploration; baseline reward of 0 means no_op never wins |
| LLM-judge gaming | T1 (programmatic) is 30% of tasks; ensemble of 3 judges in T2 with disagreement penalty |

---

## 7. OpenEnv Environment Specification

### 7.1 Scaffolding

```bash
pip install openenv-core
openenv init forge_env
cd forge_env
```

This produces the standard OpenEnv layout:

```
forge_env/
├── __init__.py           # export ForgeAction, ForgeObservation, ForgeEnv
├── models.py             # Pydantic models (we customise)
├── client.py             # ForgeEnv(EnvClient)
├── openenv.yaml          # manifest
├── README.md
└── server/
    ├── __init__.py
    ├── forge_environment.py  # OUR CORE LOGIC
    ├── app.py                # FastAPI wrapper (auto-generated)
    ├── Dockerfile
    └── requirements.txt
```

### 7.2 Action / Observation models

```python
# models.py
from pydantic import BaseModel, Field
from typing import Literal, Optional, List
from openenv.core.env_server import Action, Observation, State

class ForgeAction(Action):
    type: Literal["execute", "propose_edit", "validate", "commit", "rollback"]
    edit: Optional[dict] = None      # populated when type == propose_edit
    task_id: Optional[str] = None    # populated when type == execute

class ForgeObservation(Observation):
    phase: str                       # which phase of the loop
    playbook_snapshot: dict           # rendered playbook (trimmed for context)
    last_trajectory: Optional[dict]
    last_score: Optional[float]
    val_baseline: float
    val_after_edit: Optional[float]
    reward_components: Optional[dict]
    done: bool
    failures_buffer: List[dict]

class ForgeState(State):
    playbook_version: int
    edit_log: List[dict]
    rolling_val_accuracy: float
    rolling_complexity: int
```

### 7.3 The three OpenEnv methods

```python
# server/forge_environment.py
from openenv.core.env_server import Environment

class ForgeEnvironment(Environment):
    def __init__(self):
        super().__init__()
        self.playbook = load_initial_playbook()
        self.task_pool = build_task_pool()
        self.val_set = freeze_validation_set(seed=42)
        self.failures_buffer = []

    def reset(self) -> ForgeObservation:
        # fresh episode: load v0 playbook, sample initial failures
        self.playbook = load_initial_playbook()
        self.failures_buffer = collect_failures(
            self.executor, self.playbook, self.task_pool, n=4
        )
        return ForgeObservation(
            phase="awaiting_edit",
            playbook_snapshot=render(self.playbook),
            failures_buffer=self.failures_buffer,
            val_baseline=evaluate(self.executor, self.playbook, self.val_set),
            done=False,
        )

    def step(self, action: ForgeAction) -> ForgeObservation:
        if action.type == "propose_edit":
            new_pb = apply_edit(self.playbook, action.edit)
            if violates_budget(new_pb):
                return self._budget_violation_obs()
            val_after = evaluate(self.executor, new_pb, self.val_set)
            reward, components = compute_reward(
                self.playbook, new_pb, val_after, self.val_baseline
            )
            if reward > 0:
                self.playbook = new_pb     # commit
                self.val_baseline = val_after
            return ForgeObservation(
                phase="edit_evaluated",
                playbook_snapshot=render(self.playbook),
                val_after_edit=val_after,
                reward_components=components,
                reward=reward,
                done=(self._step_count >= MAX_EPISODE_STEPS),
            )

    @property
    def state(self) -> ForgeState:
        return ForgeState(
            playbook_version=self.playbook.version,
            edit_log=self.playbook.edit_log,
            rolling_val_accuracy=self.val_baseline,
            rolling_complexity=self.playbook.token_count(),
        )
```

### 7.4 Deployment

```bash
# Local test
uv run server
# Push to HF
openenv push --repo-id your-team/forge-env
# Confirm
pip install "forge-env @ git+https://huggingface.co/spaces/your-team/forge-env"
```

Per help-guide §13: deploy early, even before training. Catches packaging issues.

---

## 8. Sandboxing for Tool Authoring

If the Editor proposes `add_tool` or `modify_tool`, the new code path is:

1. **Static analysis** — AST scan rejects: `os.*`, `subprocess`, `eval`, `exec`, `__import__`, `open(...,"w")`, network libs
2. **Smoke test required** — Editor must propose a `pytest`-compatible test alongside the tool. Tool only accepted if test passes.
3. **Subprocess isolation** — tool runs in a separate Python process with `resource.setrlimit` (CPU=5s, memory=256MB)
4. **No FS/network** — `seccomp` filter blocks syscalls except `read`/`write` on stdin/stdout
5. **Auto-revert on crash** — any uncaught exception during validation causes the candidate to be rejected with `reward = -0.2`

### 8.1 Hackathon shortcut

For the MVP, restrict tool authoring to **a parameterised template library** (10 tool skeletons the agent can instantiate, e.g., `summarise_text(style)`, `extract_pattern(regex)`, `format_table(columns)`). This keeps the safety story simple while preserving the "agent authors tools" demo. Full open-ended authoring is a stretch goal.

---

## 9. Training Pipeline

### 9.1 Stack

- **Base model:** `Qwen/Qwen2.5-3B-Instruct` (4-bit via Unsloth)
- **Trainer:** TRL `GRPOTrainer` with `environment_factory=lambda: ForgeEnv(base_url=...)`
- **Generator backend:** Unsloth-patched HF transformers
- **Hardware target:** Single Colab T4 (16GB) — proven sufficient for Qwen3-0.6B Wordle examples; 3B is borderline but fits with Unsloth + LoRA

### 9.2 GRPO config (starting point)

```python
GRPOConfig(
    output_dir="forge_checkpoints",
    num_generations=8,                # K candidate edits per failure batch
    max_prompt_length=4096,
    max_completion_length=1024,
    learning_rate=5e-6,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    max_steps=400,                    # ~6 hours on T4
    logging_steps=5,
    save_steps=50,
    bf16=False, fp16=True,
    use_unsloth=True,
)
```

### 9.3 LoRA config

```python
LoraConfig(
    r=16, lora_alpha=32,
    target_modules=["q_proj","k_proj","v_proj","o_proj"],
    lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
)
```

### 9.4 Curriculum

Per help-guide §6, *"make success possible early."* Training proceeds in three phases:

| Phase | Steps | Tasks loaded | Purpose |
|---|---|---|---|
| **Warm-up** | 0–80 | T1 only (programmatic) | Editor learns to emit valid edit JSON; clean reward signal |
| **Diversification** | 80–250 | T1 + T2 | Editor learns rubric-judged tasks |
| **Full** | 250–400 | T1 + T2 + T3 (multi-turn sim user) | full distribution |

### 9.5 (Optional, stretch) Outer-loop Executor co-training

After a successful Editor training run, freeze the final Playbook and run a second short pass training the Executor on (Playbook, Task) → trajectory. This is the "weights catch up to scaffolding" loop. Not required for MVP.

### 9.6 Saving

Per help-guide §16: **never naïvely upcast 4-bit to 16-bit before merging LoRA**. Use Unsloth's `model.save_pretrained_merged(..., save_method="merged_16bit")` which handles this safely. Test inference immediately after saving.

---

## 10. Metrics & Evaluation — *the agent-evolution dashboard*

This is the section that wins the 30% storytelling weight. Every metric below maps to a specific plot or visualisation.

### 10.1 Primary curves (must-have)

All plotted on `training_step` x-axis with `step` units labelled.

| Metric | Y-axis | What it shows |
|---|---|---|
| **Validation accuracy** | 0.0–1.0 | The headline reward curve. Should rise. |
| **Held-out generalisation accuracy** | 0.0–1.0 | The honest curve. Plotted with val on same axes; gap = overfitting. |
| **Edit acceptance rate** | % | Fraction of proposed edits with reward > 0. Should rise then plateau. |
| **Reward components stacked** | scalar | Δval, complexity_pen, regression_pen on stacked area chart |
| **Random-baseline comparison** | 0.0–1.0 | Same x-axis, accuracy of an untrained / random-edit agent |

### 10.2 Playbook evolution metrics (must-have, the unique angle)

| Metric | Visualisation | What it shows |
|---|---|---|
| **Playbook token count over time** | Line chart | Bloat detection — should grow then stabilise |
| **Tool count over time** | Step plot | Compositional growth |
| **Lessons-written cumulative** | Line | Pace of distillation |
| **Edits by type, stacked over time** | Stacked bar | Which actions matter when |
| **Edit-attribution waterfall** | Waterfall | Each committed edit's contribution to final accuracy |
| **Regression events** | Event scatter | Edits that broke previously-passing tasks (and their rollback) |

### 10.3 Per-task-tier breakdown (must-have)

A 3-row faceted plot showing val accuracy on T1 / T2 / T3 separately. Reveals whether progress is uniform or tier-specific.

### 10.4 Qualitative artefacts (must-have for storytelling)

These go in the README and demo video:

- **Side-by-side trajectory replay** — pick 3 representative tasks, show v0-Playbook trajectory vs v_final-Playbook trajectory. Same task, different success.
- **Prompt diff gallery** — `prompts/reflector.md` v0 vs v_final, with inline annotations of what each major edit added.
- **Invented tool gallery** — every tool the agent authored, with the failure trajectory that prompted it.
- **Lessons-learned timeline** — chronological rendering of the lessons file growing.

### 10.5 Sanity checks (process-monitoring, per help-guide §15)

Logged to W&B / TensorBoard, watched during training:

- Editor JSON parse-success rate (should be ~100% after step 50; if it dips, formatting collapsed)
- Mean rollout length in tokens (should remain bounded)
- Sandboxed-tool-execution failure rate (should fall over time)
- Validation set timing (drift signals env instability)
- LLM-judge agreement rate (should stay >0.7; if it drops, judge prompt is leaking)

### 10.6 Headline number (for the README)

> *"Forge improves a Qwen2.5-3B agent's accuracy on a held-out personal-assistant task suite from **41% → 67%** over 400 GRPO steps **without updating model weights** — purely by editing its own playbook."*

(Numbers are placeholder targets; replace with actuals after training.)

### 10.7 Plot hygiene (per help guide §"make plots readable")

- Both axes labelled with units (`Training step`, `Validation accuracy (%)`)
- All key plots saved as `.png` to `outputs/plots/` and committed to repo
- Every plot has a one-line caption in the README
- Baseline and trained on the **same axes**
- W&B run linked from README

---

## 11. Visualisation & Demo

A Gradio app, deployed alongside the env on a separate HF Space.

### 11.1 Required tabs

1. **Playbook Evolution Tree** — git-like visualisation. Every commit is a node, edges show parent→child, node colour = reward delta. Click a node, see the diff.
2. **Trajectory Replay** — dropdown of held-out tasks × dropdown of playbook versions. Shows the executor's full trace side-by-side for two versions.
3. **Live Training Dashboard** — embed of W&B charts, plus current Playbook rendering.
4. **Try It Yourself** — text box: enter a personal-assistant task, watch the v_final Forge agent solve it.

### 11.2 Demo video script (90 sec)

```
0:00–0:10  "Most agents are configured by hand. We trained one to configure itself."
0:10–0:25  Show v0 Playbook → executor failing on email-reply task
0:25–0:45  Show one round of edit-and-validate happening, narrate the reward
0:45–1:10  Cut to v_final Playbook → executor succeeding on the same task
1:10–1:25  Pan over playbook evolution tree, highlight 2 invented tools
1:25–1:30  Reward curve + headline number + URL
```

---

## 12. Implementation Roadmap

### 12.1 Pre-hackathon (next 3 days)

- [ ] Clone OpenEnv repo, run echo_env locally end-to-end (validates dev environment)
- [ ] Run TRL Wordle GRPO Colab unmodified (validates training stack)
- [ ] Pick base model finalised (Qwen2.5-3B-Instruct vs Qwen3-0.6B fallback)
- [ ] HF account + Space pre-created
- [ ] Pick a task generator library (`faker` for names, `random` for windows)

### 12.2 Day 1 (onsite, ~10 hours)

| Time | Owner A (Env) | Owner B (Rewards) | Owner C (Training) | Owner D (Demo) |
|---|---|---|---|---|
| 0–2h | `openenv init forge_env`; implement Playbook class + apply_edit | Implement T1 task generator + grader | Set up Colab with Unsloth + Qwen | Sketch Gradio layout |
| 2–4h | Implement reset/step/state | Implement T2 (rubric judge) | Stub GRPO loop end-to-end | Build playbook-tree component |
| 4–6h | Local end-to-end smoke test | Wire 3 reward components | Train on 1 sample edit, confirm gradient flow | Trajectory replay component |
| 6–8h | `openenv push` to Spaces | Anti-hack tests (memorisation, bloat) | First real training run, 50 steps | Prompt diff viewer |
| 8–10h | Fix what broke when deployed | Calibrate λ_complexity, λ_regression | Inspect generations; check no reward hacking | Hook live W&B |

End-of-day-1 deliverable: env is on HF Spaces, training runs end-to-end on T1 only, val accuracy moves above baseline.

### 12.3 Day 2 (onsite, ~10 hours)

| Time | Activity |
|---|---|
| 0–2h | Add T2 tasks, retrain warm-up→diversification |
| 2–4h | Add `add_tool` action with template-restricted authoring |
| 4–6h | Full curriculum training run (400 steps) |
| 6–8h | Generate all required plots; record demo video |
| 8–9h | Write README + blog post |
| 9–10h | Submit |

### 12.4 MVP / stretch breakdown

| Feature | MVP (must) | Stretch | Cut if behind |
|---|---|---|---|
| Programmatic tasks (T1) | ✅ | | |
| Rubric-judged tasks (T2) | ✅ | | |
| Sim-user tasks (T3) | | ✅ | ✅ cut |
| `edit_prompt` action | ✅ | | |
| `write_lesson` action | ✅ | | |
| `add_tool` (template-restricted) | ✅ | | |
| `add_tool` (open Python authoring) | | ✅ | ✅ cut |
| `edit_routing` action | | ✅ | ✅ cut |
| 3-term reward | ✅ | | |
| Δ-validation only | ✅ fallback | | (use this if regression-pen breaks) |
| Playbook evolution tree UI | ✅ | | |
| Edit-attribution waterfall | | ✅ | ✅ cut |
| Live W&B embed | ✅ | | |
| HF Space deployment | ✅ | | |
| Demo video | ✅ | | |
| Blog post | ✅ | | |

---

## 13. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| GRPO doesn't converge in 400 steps | M | H | Start curriculum on T1 only; have a fallback to Qwen3-0.6B with simpler action space |
| Editor outputs malformed JSON | M | M | Schema validation + format-rewards; warm-up phase explicitly trains formatting |
| Validation eval is too slow (rollouts dominate) | H | H | Cap val set at 40; cache executor outputs by (playbook_hash, task_id); use vLLM if available |
| Reward hacking via prompt bloat | M | M | Hard token cap + complexity penalty |
| Sandboxing escape via authored tool | L | H | MVP uses template-restricted tools only |
| HF Space deployment fails on demo day | L | H | Deploy by end of Day 1; have local Docker fallback |
| LoRA save corrupts model | L | M | Test inference immediately after first save |
| Single judge bias in T2 | M | M | Ensemble of 3 judges; T1 is 30% of distribution |

---

## 14. Submission Checklist

Per the official judging brief:

- [ ] OpenEnv environment compliant with latest release, tested locally
- [ ] Environment hosted on Hugging Face Spaces, URL in README
- [ ] Training script (Colab notebook) using TRL + Unsloth, runnable by judges
- [ ] Reward and loss plots saved as `.png` in `outputs/plots/`, embedded in README
- [ ] W&B run link in README
- [ ] Baseline-vs-trained comparison plot on same axes
- [ ] Mini-blog on Hugging Face OR <2 min YouTube video, linked from README
- [ ] README explains: problem, environment, agent behaviour, results, why it matters (3–5 minute read)
- [ ] No reserved tool names (`reset`, `step`, `state`, `close`)
- [ ] `openenv.yaml` manifest valid
- [ ] No huge video files in the repo (link externally)

---

## 15. File-Tree Summary (final repo)

```
forge/
├── README.md                       # 3–5 min read: problem → env → results
├── BLOG.md                         # mini-blog content (also pushed to HF Hub)
├── forge_env/                      # the OpenEnv environment
│   ├── __init__.py
│   ├── models.py                   # ForgeAction, ForgeObservation, ForgeState
│   ├── client.py
│   ├── openenv.yaml
│   ├── playbook/                   # initial v0 playbook
│   │   ├── manifest.yaml
│   │   ├── prompts/
│   │   ├── tools/builtin/
│   │   └── lessons/
│   └── server/
│       ├── forge_environment.py    # core env logic
│       ├── app.py
│       ├── tasks/
│       │   ├── tier1_programmatic.py
│       │   ├── tier2_rubric.py
│       │   └── tier3_simuser.py
│       ├── rewards/
│       │   ├── delta_validation.py
│       │   ├── complexity.py
│       │   └── regression.py
│       ├── sandbox/
│       │   ├── ast_check.py
│       │   └── subprocess_runner.py
│       ├── Dockerfile
│       └── requirements.txt
├── training/
│   ├── train_forge.ipynb           # main Colab
│   ├── train_forge.py              # script form
│   ├── grpo_config.py
│   └── unsloth_setup.py
├── demo/
│   ├── app.py                      # Gradio
│   └── components/
├── outputs/
│   ├── plots/
│   │   ├── 01_val_accuracy.png
│   │   ├── 02_holdout_generalisation.png
│   │   ├── 03_reward_components_stacked.png
│   │   ├── 04_playbook_complexity.png
│   │   ├── 05_per_tier_breakdown.png
│   │   ├── 06_edit_attribution_waterfall.png
│   │   └── 07_baseline_vs_trained.png
│   ├── playbook_versions/          # one .yaml per major checkpoint
│   └── trajectories/               # for replay
└── tests/
    ├── test_apply_edit.py
    ├── test_reward_components.py
    └── test_sandbox.py
```

---

## 16. Appendices

### Appendix A — Sample task instances (one per tier)

**T1 (programmatic):**
```
Task: Schedule a 60-minute meeting between Alice, Bob, and Carol next week.
Constraints: Alice is busy Mon-Tue all day. Bob is free only Wed 1-3pm and Fri 9-11am.
Carol is free Wed-Fri afternoons. Find the earliest valid slot.
Expected output: {"day": "Wednesday", "start": "13:00", "end": "14:00"}
Grader: hard equality on the JSON
```

**T2 (rubric-judged):**
```
Task: Reply to this email from your manager: "I'm noticing the quarterly report
keeps slipping. Can you tell me what's going on?"
Hidden persona: User is a senior IC who delivered late because of an unannounced
dependency from another team. Wants to be honest but professional, not defensive.
Rubric: acknowledged the slip / named the dependency / proposed a path forward /
matched neutral-professional tone / no grovelling / no blame-throwing
Grader: ensemble of 3 LLM judges, mean of 6-dim rubric
```

**T3 (sim-user):**
```
Task: Help the user (a simulated picky shopper) pick a laptop under $1500 for
video editing. Discover their priorities through dialogue, propose 2 candidates,
finalise a choice.
Hidden persona goals: (1) at least 32GB RAM (will reject anything less),
(2) prefers AMD over Intel (mild), (3) battery life >8h (will not say upfront)
Grader: did final pick satisfy all hard constraints? did agent ask <=4 questions?
```

### Appendix B — Sample committed edit

```yaml
version: 23
parent: 22
edit:
  type: edit_prompt
  target: prompts/reflector.md
  diff: |
    @@ -3,5 +3,9 @@
     Before producing the final answer:
     1. Re-read the original task description.
     2. List every constraint mentioned by the user.
    +3. For each constraint, identify exactly which step in your trajectory
    +   addressed it. If any constraint is unaddressed, plan one more action.
    +4. If the user used hedging words ("maybe", "if possible"), treat them
    +   as soft preferences not hard constraints.
     5. Format the final answer per the formatter template.
  rationale: "Multiple T2 failures showed reflector ignoring late-introduced
              constraints in long emails."
reward_components:
  delta_val: +0.092
  complexity: -0.011 (28 new tokens)
  regression: 0.0
  total: +0.081
```

### Appendix C — Glossary

- **Playbook** — versioned bundle of prompts/tools/lessons that defines agent behaviour
- **Engine** — LLM weights + inference loop
- **Executor** — Engine running in execution mode, given (Task, Playbook)
- **Editor** — Engine running in editor mode, given (Failure, Playbook)
- **Trajectory** — record of one Executor rollout (states, actions, observations, score)
- **Edit action** — one of six structured operations the Editor can emit
- **Validation set** — fixed task slice used to compute Δ-validation reward
- **Held-out set** — separate task slice using unseen templates, for honest generalisation evaluation
- **Edit-attribution** — post-hoc analysis of how much each committed edit contributed to final accuracy
- **GRPO** — Group Relative Policy Optimisation, the TRL trainer we use

---

*End of PRD v1.0. Open issues for changes; do not edit in place.*
