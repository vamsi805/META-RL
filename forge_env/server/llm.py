from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Protocol


Message = Dict[str, str]


class LLMClient(Protocol):
    def generate(
        self,
        messages: List[Message],
        *,
        max_new_tokens: int = 512,
        temperature: float = 0.2,
    ) -> str:
        ...


@dataclass(frozen=True)
class LLMSettings:
    backend: str
    model_name: str
    load_in_4bit: bool = False


class TransformersLLMClient:
    def __init__(self, model_name: str, load_in_4bit: bool = False) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "Transformers backend needs torch and transformers installed. "
                "Install with `pip install -e \".[train]\" torch accelerate`."
            ) from exc

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        kwargs: Dict[str, Any] = {
            "device_map": "auto",
            "trust_remote_code": True,
        }
        if load_in_4bit:
            kwargs["load_in_4bit"] = True
        else:
            kwargs["torch_dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)

    def generate(
        self,
        messages: List[Message],
        *,
        max_new_tokens: int = 512,
        temperature: float = 0.2,
    ) -> str:
        prompt = self._format_messages(messages)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        do_sample = temperature > 0
        output = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=max(temperature, 1e-5),
            do_sample=do_sample,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        generated = output[0][inputs["input_ids"].shape[-1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

    def _format_messages(self, messages: List[Message]) -> str:
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        parts = []
        for msg in messages:
            parts.append(f"{msg['role'].upper()}:\n{msg['content']}")
        parts.append("ASSISTANT:\n")
        return "\n\n".join(parts)


class HuggingFaceApiLLMClient:
    def __init__(self, model_name: str) -> None:
        try:
            from huggingface_hub import InferenceClient
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "Hugging Face API backend needs huggingface_hub installed. "
                "Install with `pip install huggingface_hub`."
            ) from exc
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
        self.client = InferenceClient(model=model_name, token=token)

    def generate(
        self,
        messages: List[Message],
        *,
        max_new_tokens: int = 512,
        temperature: float = 0.2,
    ) -> str:
        response = self.client.chat_completion(
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=temperature,
        )
        choice = response.choices[0]
        return choice.message.content or ""


def llm_settings_from_env(
    *,
    backend: str | None = None,
    model_name: str | None = None,
) -> LLMSettings:
    return LLMSettings(
        backend=(backend or os.environ.get("FORGE_LLM_BACKEND", "stub")).lower(),
        model_name=model_name
        or os.environ.get("FORGE_LLM_MODEL", "Qwen/Qwen2.5-1.5B-Instruct"),
        load_in_4bit=os.environ.get("FORGE_LLM_LOAD_IN_4BIT", "0") == "1",
    )


@lru_cache(maxsize=4)
def get_llm_client(
    backend: str,
    model_name: str,
    load_in_4bit: bool = False,
) -> LLMClient | None:
    if backend in {"", "stub", "none"}:
        return None
    if backend == "transformers":
        return TransformersLLMClient(model_name, load_in_4bit=load_in_4bit)
    if backend in {"hf_api", "huggingface_api"}:
        return HuggingFaceApiLLMClient(model_name)
    raise ValueError(f"Unknown FORGE_LLM_BACKEND={backend!r}")


def get_configured_llm_client(
    *,
    backend: str | None = None,
    model_name: str | None = None,
) -> LLMClient | None:
    settings = llm_settings_from_env(backend=backend, model_name=model_name)
    return get_llm_client(settings.backend, settings.model_name, settings.load_in_4bit)


def extract_json_object(text: str) -> Dict[str, Any] | None:
    """Best-effort parser for model replies that should contain one JSON object."""
    text = text.strip()
    if not text:
        return None
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fenced:
        try:
            data = json.loads(fenced.group(1))
            return data if isinstance(data, dict) else None
        except json.JSONDecodeError:
            pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            data = json.loads(text[start : end + 1])
            return data if isinstance(data, dict) else None
        except json.JSONDecodeError:
            return None
    return None
