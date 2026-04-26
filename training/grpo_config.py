from dataclasses import dataclass


@dataclass
class ForgeTrainConfig:
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    llm_backend: str = "stub"
    max_steps: int = 400
    smoke_steps: int = 8
    learning_rate: float = 5e-6
    num_generations: int = 8
    max_prompt_length: int = 4096
    max_completion_length: int = 1024
