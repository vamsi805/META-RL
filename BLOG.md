# Forge MVP Build Notes

Forge demonstrates self-improvement without changing model weights by
optimizing a structured Playbook artifact through edit-and-validate cycles.

## MVP Highlights

- OpenEnv-like environment contract with `reset`, `step`, and `state`.
- Versioned Playbook with budget checks and reward-aware edit commits.
- Procedural T1 tasks with deterministic grading.
- Rubric-based T2 tasks with pluggable judge backend and local fallback.
- Model-agnostic training script with smoke mode.

## Next Steps

- Add full T3 simulated-user interactions.
- Add open-ended tool authoring sandbox hardening.
- Run longer training and publish full benchmark plots.
