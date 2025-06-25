FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

WORKDIR /app

COPY uv.lock .
COPY pyproject.toml .

RUN uv sync

COPY src ./src

CMD ["uv", "run", "src/main.py"]
