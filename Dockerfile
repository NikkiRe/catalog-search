FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml uv.lock README.md LICENSE THIRD_PARTY_NOTICES.md ./
COPY src ./src
COPY web ./web
RUN pip install --no-cache-dir uv && uv sync --locked --no-editable --no-dev
ENV PATH="/app/.venv/bin:$PATH"
COPY scripts ./scripts
COPY licenses ./licenses
COPY reports ./reports
RUN python scripts/download_demo.py
EXPOSE 8765
CMD ["catalog-search", "serve", "--host", "0.0.0.0", "--port", "8765", "--artifacts", "/app/artifacts"]
