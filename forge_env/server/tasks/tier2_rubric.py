from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from typing import List, Protocol

from forge_env.server.llm import extract_json_object, get_configured_llm_client


@dataclass
class Tier2Task:
    task_id: str
    prompt: str
    rubric: List[str]
    tier: str = "T2"
    task_type: str = "email"
    template_family: str = "alpha"


class JudgeBackend(Protocol):
    def score(self, response: str, rubric: List[str]) -> float:
        ...


class HeuristicJudge:
    """Single judge; ``mode`` varies scoring strictness for ensemble diversity."""

    def __init__(self, mode: str = "normal") -> None:
        self.mode = mode

    def score(self, response: str, rubric: List[str]) -> float:
        lower = response.lower()
        if not rubric:
            return 0.0
        hits = 0
        for item in rubric:
            words = [w.lower() for w in item.split() if len(w) > 1]
            if self.mode == "strict":
                need = words[:2] if len(words) >= 2 else words
                if need and all(w in lower for w in need):
                    hits += 1
            elif self.mode == "loose":
                if words and any(w in lower for w in words):
                    hits += 1
            else:
                token = words[0] if words else ""
                if token and token in lower:
                    hits += 1
        return hits / len(rubric)


class LLMRubricJudge:
    """Optional model judge for written responses."""

    def score(self, response: str, rubric: List[str]) -> float:
        client = get_configured_llm_client()
        if client is None:
            return HeuristicJudge("normal").score(response, rubric)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a strict grading judge. Return only JSON with a score "
                    "between 0 and 1 and a short reason. Shape: "
                    '{"score":0.0,"reason":"..."}'
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {"response": response, "rubric": rubric},
                    ensure_ascii=False,
                ),
            },
        ]
        raw = client.generate(messages, max_new_tokens=200, temperature=0.0)
        data = extract_json_object(raw)
        try:
            score = float((data or {}).get("score", 0.0))
        except (TypeError, ValueError):
            score = 0.0
        return max(0.0, min(1.0, score))


T2_TEMPLATES_ALPHA = [
    "Reply to your manager about a delayed report professionally.",
    "Draft a polite refusal email for an unrealistic deadline.",
    "Summarize a long project thread with action items.",
]
T2_TEMPLATES_BETA = [
    "Your client asks for a discount after scope creep; respond diplomatically.",
    "Write a follow-up after missing a stand-up without over-apologizing.",
    "Decline a meeting invite that conflicts with deep-work block, propose an alternative.",
]
DEFAULT_RUBRIC = [
    "acknowledge issue",
    "state cause",
    "propose plan",
    "maintain professional tone",
]


def generate_tier2(
    seed: int,
    count: int,
    family: str = "alpha",
) -> List[Tier2Task]:
    rng = random.Random(seed)
    templates = T2_TEMPLATES_ALPHA if family == "alpha" else T2_TEMPLATES_BETA
    tasks: List[Tier2Task] = []
    for idx in range(count):
        tasks.append(
            Tier2Task(
                task_id=f"t2_{family}_{seed}_{idx}",
                prompt=rng.choice(templates),
                rubric=list(DEFAULT_RUBRIC),
                template_family=family,
            )
        )
    return tasks


def tier2_task_to_dict(t: Tier2Task) -> dict:
    return {
        "task_id": t.task_id,
        "prompt": t.prompt,
        "rubric": t.rubric,
        "tier": t.tier,
        "task_type": t.task_type,
        "template_family": t.template_family,
    }


def grade_tier2(response: str, rubric: List[str], judge: JudgeBackend | None = None) -> float:
    backend = judge or HeuristicJudge("normal")
    return backend.score(response, rubric)


def grade_tier2_ensemble(response: str, rubric: List[str]) -> float:
    if os.environ.get("FORGE_LLM_JUDGE", "0") == "1":
        return LLMRubricJudge().score(response, rubric)

    """Three heuristic judges with different strictness (PRD: reduce single-judge gaming)."""
    scores = [
        HeuristicJudge("normal").score(response, rubric),
        HeuristicJudge("strict").score(response, rubric),
        HeuristicJudge("loose").score(response, rubric),
    ]
    return sum(scores) / len(scores)
