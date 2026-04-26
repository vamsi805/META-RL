"""Optional TRL + PyTorch path (P3-adjacent): not required for the default training loop.

The default ``train_forge.py`` implements a GRPO-style **group scoring** pass over
candidate edits using ``ForgeEnvironment.preview_propose``. Full ``GRPOTrainer``
fine-tuning needs ``trl``, ``torch``, a GPU, and a remote or in-process policy;
see Hugging Face TRL OpenEnv docs when you are ready to attach a real model.
"""

from __future__ import annotations


def describe_trl_status(model_name: str) -> None:
    try:
        import torch  # noqa: F401
        _ = torch.__version__
        has_torch = True
    except ImportError:
        has_torch = False
    try:
        import trl  # noqa: F401

        has_trl = True
    except ImportError:
        has_trl = False
    _ = model_name
    if not has_torch or not has_trl:
        pass  # informational only; avoid printing in library mode
