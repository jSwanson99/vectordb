FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

WORKDIR /app

COPY uv.lock .
COPY pyproject.toml .

RUN uv sync

COPY main.py .

CMD ["uv", "run", "main.py"]
