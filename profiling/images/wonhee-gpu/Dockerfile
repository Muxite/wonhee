# wonhee + the `gpu` dependency group (torch cu130, transformers, demucs), for the GPU bench (python -m wonhee.bench).
# Built with the repo root as context:  docker build -f profiling/images/wonhee-gpu/Dockerfile .
# The torch wheels carry their own CUDA runtime, so a slim base works; the driver comes from the nvidia runtime.
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg git ca-certificates libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.11.7 /uv /usr/local/bin/uv
WORKDIR /opt/wonhee
ENV NANOBOT_SKIP_WEBUI_BUILD=1 UV_LINK_MODE=copy
# dependencies first, in their own layer: a code change then rebuilds in seconds instead of reinstalling torch
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --group gpu --no-install-project && rm -rf /root/.cache/uv
COPY wonhee ./wonhee
COPY gauge ./gauge
RUN uv sync --frozen --no-dev --group gpu && rm -rf /root/.cache/uv
ENV PATH=/opt/wonhee/.venv/bin:$PATH WONHEE_HOME=/tmp/wonhee HF_HUB_OFFLINE=1 GAUGE=1
