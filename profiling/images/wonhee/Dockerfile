# Built with the wonhee repo as context:  docker build -f images/wonhee/Dockerfile ../wonhee
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg git ca-certificates \
 && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.11.7 /uv /usr/local/bin/uv
WORKDIR /opt/wonhee
COPY pyproject.toml uv.lock README.md ./
COPY wonhee ./wonhee
ENV NANOBOT_SKIP_WEBUI_BUILD=1 UV_LINK_MODE=copy
RUN uv sync --frozen --no-dev && rm -rf /root/.cache/uv
ENV PATH=/opt/wonhee/.venv/bin:$PATH WONHEE_HOME=/tmp/wonhee HF_HUB_OFFLINE=1
