ARG BASE_IMAGE=ghcr.io/meta-pytorch/openenv-base:latest
FROM ${BASE_IMAGE}

# Install dependencies
COPY server/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

# Copy environment code — HF Space uploads env folder contents to root
COPY . /app/envs/agent_safety_env/

WORKDIR /app

ENV PYTHONPATH=/app/src:/app/envs
ENV MAX_CONCURRENT_ENVS=8
ENV ENABLE_WEB_INTERFACE=true
ENV API_BASE_URL=https://router.huggingface.co/v1
ENV MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
# HF_TOKEN is injected at runtime via HF Spaces secrets — do not hardcode here

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["uvicorn", "envs.agent_safety_env.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
