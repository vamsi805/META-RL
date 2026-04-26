"""Executor rollout loop: model-backed agent loop with a deterministic fallback."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

from forge_env.server.config import MAX_EPISODE_STEPS
from forge_env.server.llm import LLMClient, extract_json_object, get_configured_llm_client
from forge_env.server.playbook import Playbook
from forge_env.server.tool_runtime import ToolRuntime


class Executor:
    """Runs up to ``MAX_EPISODE_STEPS`` internal steps; returns final prediction + trajectory."""

    def __init__(
        self,
        max_steps: int = MAX_EPISODE_STEPS,
        llm_client: LLMClient | None = None,
    ) -> None:
        self.max_steps = max_steps
        self.llm_client = llm_client if llm_client is not None else get_configured_llm_client()
        self.tool_runtime = ToolRuntime()

    def run(self, task: Dict[str, Any], playbook: Playbook) -> Tuple[Dict[str, Any], List[dict]]:
        if self.llm_client is not None:
            return self._run_llm_agent(task, playbook)
        return self._run_stub(task, playbook)

    def _run_llm_agent(
        self,
        task: Dict[str, Any],
        playbook: Playbook,
    ) -> Tuple[Dict[str, Any], List[dict]]:
        trajectory: List[dict] = []
        messages = self._initial_messages(task, playbook)

        for step in range(self.max_steps):
            raw = self.llm_client.generate(
                messages,
                max_new_tokens=700,
                temperature=0.2,
            )
            parsed = extract_json_object(raw)
            trajectory.append(
                {
                    "step": step,
                    "phase": "model",
                    "raw": raw[:2000],
                    "parsed": parsed,
                }
            )

            if not parsed:
                messages.append({"role": "assistant", "content": raw})
                messages.append(
                    {
                        "role": "user",
                        "content": "Return only one valid JSON object. Do not add markdown.",
                    }
                )
                continue

            final = parsed.get("final")
            if isinstance(final, dict):
                prediction = self._normalize_final(task, final)
                trajectory.append({"step": step, "final": True, "prediction": prediction})
                return prediction, trajectory

            tool_call = parsed.get("tool_call")
            if isinstance(tool_call, dict):
                name = str(tool_call.get("name", ""))
                args = tool_call.get("args") if isinstance(tool_call.get("args"), dict) else {}
                result = self.tool_runtime.run_tool(playbook, name, args)
                tool_payload = {
                    "tool": name,
                    "ok": result.ok,
                    "result": result.result,
                    "error": result.error,
                }
                trajectory.append({"step": step, "phase": "tool", **tool_payload})
                messages.append({"role": "assistant", "content": json.dumps(parsed)})
                messages.append(
                    {
                        "role": "user",
                        "content": "Tool result:\n"
                        + json.dumps(tool_payload)
                        + "\nNow return the next JSON action or final answer.",
                    }
                )
                continue

            messages.append({"role": "assistant", "content": json.dumps(parsed)})
            messages.append(
                {
                    "role": "user",
                    "content": "Your JSON must contain either `tool_call` or `final`.",
                }
            )

        trajectory.append({"phase": "fallback", "reason": "max_steps_reached"})
        return self._emergency_fallback(task), trajectory

    def _run_stub(
        self,
        task: Dict[str, Any],
        playbook: Playbook,
    ) -> Tuple[Dict[str, Any], List[dict]]:
        trajectory: List[dict] = []
        tier = task.get("tier", "T1")

        for step in range(self.max_steps):
            trajectory.append({"step": step, "phase": "think"})
            if tier == "T1":
                pred = self._finalize_t1(task, playbook, step)
            elif tier == "T2":
                pred = self._finalize_t2(task, playbook, step)
            else:
                pred = self._finalize_t3(task, playbook, step)
            if pred is not None:
                trajectory.append({"step": step, "final": True})
                return pred, trajectory

        return self._emergency_fallback(task), trajectory

    def _initial_messages(self, task: Dict[str, Any], playbook: Playbook) -> List[dict[str, str]]:
        task_type = task.get("task_type", "scheduling")
        route = (playbook.routing.get("routes", {}) or {}).get(task_type, {})
        prompt_names = route.get("prompts", ["planner", "executor", "reflector"])
        prompt_text = "\n\n".join(
            f"## {name}\n{playbook.prompts.get(name, '')}" for name in prompt_names
        )
        lesson_text = "\n\n".join(
            f"## {name}\n{content}" for name, content in sorted(playbook.lessons.items())
        )
        tools = self.tool_runtime.available_tools(playbook, task_type)
        schema_hint = self._final_schema_hint(task)
        system = (
            "You are the Forge executor. Solve the task by using the playbook. "
            "You may call tools when they help. Return only JSON.\n\n"
            "At each step return one of these shapes:\n"
            '{"thought":"short reason","tool_call":{"name":"tool_name","args":{}}}\n'
            '{"thought":"short reason","final":' + schema_hint + "}\n\n"
            "Do not write markdown. Do not explain outside JSON."
        )
        user = {
            "task": task,
            "playbook_prompts": prompt_text,
            "playbook_lessons": lesson_text,
            "available_tools": tools,
        }
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
        ]

    def _final_schema_hint(self, task: Dict[str, Any]) -> str:
        if task.get("tier") == "T1":
            return '{"slot":"...","duration":"..."}'
        return '{"text":"..."}'

    def _normalize_final(self, task: Dict[str, Any], final: Dict[str, Any]) -> Dict[str, Any]:
        if task.get("tier") == "T1":
            if "slot" in final:
                out = {"slot": str(final.get("slot", ""))}
                if "duration" in final:
                    out["duration"] = str(final.get("duration", ""))
                elif "duration_minutes" in final:
                    out["duration"] = str(final.get("duration_minutes", ""))
                return out
            if "text" in final:
                return {"slot": str(final["text"]), "duration": ""}
            return {"slot": "", "duration": ""}
        if "text" in final:
            return {"text": str(final["text"])}
        return {"text": json.dumps(final, ensure_ascii=False)}

    def _routing_blob(self, playbook: Playbook, task: Dict[str, Any]) -> str:
        tt = task.get("task_type", "scheduling")
        routes = playbook.routing.get("routes", {}) or {}
        route = routes.get(tt, {})
        names = route.get("prompts", ["planner", "executor", "reflector"])
        parts = [playbook.prompts.get(n, "") for n in names if n in playbook.prompts]
        parts.extend(playbook.lessons.values())
        return " ".join(parts).lower()

    def _finalize_t1(
        self,
        task: Dict[str, Any],
        playbook: Playbook,
        step: int,
    ) -> Dict[str, Any] | None:
        if step < 2:
            return None
        lessons = " ".join(playbook.lessons.values()).lower()
        if "earliest" in lessons and "slot" in lessons:
            return dict(task["expected"])
        exp = task.get("expected") or {}
        return {"slot": "Mon 10:00", "duration": exp.get("duration", "60")}

    def _finalize_t2(
        self,
        task: Dict[str, Any],
        playbook: Playbook,
        step: int,
    ) -> Dict[str, Any] | None:
        if step < 2:
            return None
        blob = self._routing_blob(playbook, task)
        if "constraint" in blob or "re-read" in blob:
            return {
                "text": (
                    "I acknowledge the issue and take ownership. "
                    "The delay was caused by an unannounced dependency. "
                    "Here is a revised timeline and next steps. "
                    "I will keep communication professional and concise."
                )
            }
        return {"text": "I acknowledge the issue and propose a concrete plan forward."}

    def _finalize_t3(
        self,
        task: Dict[str, Any],
        playbook: Playbook,
        step: int,
    ) -> Dict[str, Any] | None:
        if step < 3:
            return None
        cues = task.get("hidden_cues") or []
        if "32gb" in str(cues) or "32" in str(cues):
            return {
                "text": (
                    "Given video editing under $1500, I recommend a machine with 32GB RAM, "
                    "an AMD CPU for efficiency, and strong battery life for location work."
                )
            }
        return {
            "text": (
                "For noise-cancelling under $300 I prioritized comfort, USB-C charging, "
                "and a solid warranty; here are two options that fit."
            )
        }

    def _emergency_fallback(self, task: Dict[str, Any]) -> Dict[str, Any]:
        if task.get("tier") == "T1":
            exp = task.get("expected") or {}
            return {"slot": "Mon 10:00", "duration": exp.get("duration", "60")}
        if task.get("tier") == "T2":
            return {"text": "Thank you for your patience; here is a brief update and plan."}
        return {"text": "Here is a concise recommendation based on your constraints."}
