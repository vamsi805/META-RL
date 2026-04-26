# OpenEnv-style container: environment server runs isolated from the host.
# Tool code executed during validation uses an additional subprocess with
# resource limits; for stricter isolation run this image with --read-only --network none.
FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip

COPY pyproject.toml README.md ./
COPY forge_env ./forge_env

RUN pip install --no-cache-dir .

EXPOSE 8000

CMD ["uvicorn", "forge_env.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
