import os

# Fast, CI-friendly tool runs (no Docker pull / daemon required).
os.environ.setdefault("FORGE_TOOL_ISOLATION", "subprocess")
