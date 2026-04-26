"""Training curriculum: which task tiers are active at a given global step (PRD §9.4)."""

from __future__ import annotations

WARMUP_END = 80
DIVERSIFY_END = 250


def active_tiers_for_step(step: int) -> set[str]:
    if step < WARMUP_END:
        return {"T1"}
    if step < DIVERSIFY_END:
        return {"T1", "T2"}
    return {"T1", "T2", "T3"}


def phase_name(step: int) -> str:
    if step < WARMUP_END:
        return "warmup"
    if step < DIVERSIFY_END:
        return "diversification"
    return "full"
