FROM python:3.11-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for layer caching
COPY agent_safety_env/server/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

# Copy the full environment package
COPY . /app/

ENV PYTHONPATH=/app
ENV MAX_CONCURRENT_ENVS=8
ENV ENABLE_WEB_INTERFACE=true
ENV ALLOW_SEED_FALLBACK=1
ENV API_BASE_URL=https://router.huggingface.co/v1
ENV MODEL_NAME=Qwen/Qwen2.5-72B-Instruct

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["uvicorn", "agent_safety_env.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
