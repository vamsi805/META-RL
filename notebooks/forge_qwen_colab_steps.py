# %% [markdown]
# # Forge Qwen Colab steps
#
# Run these cells in order on a Colab T4 GPU. Start with the small Qwen model first.

# %%
import subprocess

subprocess.run("nvidia-smi", shell=True, check=False)

# %% [markdown]
# Clone your repo. Replace the URL with your real repo URL if needed.

# %%
import os
import subprocess

subprocess.run("git clone https://github.com/YOUR_NAME/META-RL.git", shell=True, check=True)
os.chdir("META-RL")

# %%
import subprocess

subprocess.run("pip install -U pip", shell=True, check=True)
subprocess.run('pip install -e ".[train,dev]"', shell=True, check=True)
subprocess.run(
    "pip install -U torch --index-url https://download.pytorch.org/whl/cu121",
    shell=True,
    check=True,
)

# %% [markdown]
# Log in if you need gated models or want to push outputs to Hugging Face.

# %%
from huggingface_hub import login

login()

# %% [markdown]
# Small Qwen model. Use this first.

# %%
import os

os.environ["FORGE_LLM_BACKEND"] = "transformers"
os.environ["FORGE_LLM_MODEL"] = "Qwen/Qwen2.5-1.5B-Instruct"
os.environ["FORGE_LLM_LOAD_IN_4BIT"] = "0"
os.environ["FORGE_TOOL_ISOLATION"] = "subprocess"
os.environ["FORGE_LLM_JUDGE"] = "0"

# %% [markdown]
# Larger Qwen model on T4. Run this cell instead of the small model cell if you want 7B.

# %%
import os

os.environ["FORGE_LLM_BACKEND"] = "transformers"
os.environ["FORGE_LLM_MODEL"] = "Qwen/Qwen2.5-7B-Instruct"
os.environ["FORGE_LLM_LOAD_IN_4BIT"] = "1"
os.environ["FORGE_TOOL_ISOLATION"] = "subprocess"
os.environ["FORGE_LLM_JUDGE"] = "0"

# %% [markdown]
# Tiny real model run.

# %%
import os
import subprocess

subprocess.run(
    "python training/train_forge.py "
    "--llm_backend transformers "
    f"--model_name {os.environ['FORGE_LLM_MODEL']} "
    "--max_steps 3",
    shell=True,
    check=True,
)

# %% [markdown]
# Longer run after the small run works.

# %%
import os
import subprocess

subprocess.run(
    "python training/train_forge.py "
    "--llm_backend transformers "
    f"--model_name {os.environ['FORGE_LLM_MODEL']} "
    "--max_steps 25",
    shell=True,
    check=True,
)

# %%
import subprocess

subprocess.run("python outputs/plots/generate_plots.py", shell=True, check=True)

# %%
import subprocess

from google.colab import files

subprocess.run("zip -r forge_outputs.zip outputs", shell=True, check=True)
files.download("forge_outputs.zip")

# %% [markdown]
# Push outputs to a Hugging Face dataset repo.

# %%
from huggingface_hub import HfApi, upload_folder

api = HfApi()
repo_id = "YOUR_NAME/forge-experiment-outputs"
api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)

upload_folder(
    repo_id=repo_id,
    repo_type="dataset",
    folder_path="outputs",
    path_in_repo="outputs",
)
