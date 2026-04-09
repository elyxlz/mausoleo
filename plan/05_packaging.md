# Phase 5: Open Source Packaging

## Goal

Package the project for public release. Uses `uv` throughout for dependency management, running scripts, and builds. Clean repo structure, documentation, and easy setup.

## 5.1 Package Name

`mausoleo` — short, memorable, references the project origin.

(Or a more generic name if preferred — but given that this is newspaper-specific for now, a distinctive name is fine.)

## 5.2 Repo Structure

```
mausoleo/
  README.md
  LICENSE
  pyproject.toml
  CLAUDE.md

  plan/                          # this planning directory (keep or remove for release)

  src/mausoleo/
    __init__.py
    cli.py                       # typer CLI entry point

    ocr/
      __init__.py
      pipeline.py                # Ray Data OCR pipeline
      prompts.py                 # OCR and cleanup prompt templates
      preprocessing.py           # image preprocessing (if needed)
      models.py                  # dataclasses for OCR output

    index/
      __init__.py
      pipeline.py                # Ray Data summarization pipeline
      prompts.py                 # summarization prompt templates
      schema.py                  # ClickHouse schema definitions
      models.py                  # dataclasses for tree nodes
      writer.py                  # write nodes to ClickHouse
      embeddings.py              # embedding generation

    server/
      __init__.py
      app.py                     # FastAPI application
      routes.py                  # API endpoints
      search.py                  # search logic (semantic, text, hybrid)
      db.py                      # ClickHouse client wrapper

    eval/
      __init__.py
      metrics.py                 # CER, WER, reading order metrics
      evaluate.py                # run evaluation
      compare.py                 # comparison reporting

  eval/
    ground_truth/                # ground truth data (maybe separate repo / release artifact)
    scripts/

  tests/
    test_ocr/
    test_index/
    test_server/
    test_cli/
```

## 5.3 pyproject.toml

Uses `uv` for all dependency management. All commands use `uv run`, `uv add`, `uv sync`.

```toml
[project]
name = "mausoleo"
version = "0.1.0"
description = "Hierarchical knowledge index for historical newspaper archives"
requires-python = ">=3.11"
dependencies = [
    "typer",
    "httpx",
    "fastapi",
    "uvicorn",
    "pydantic>=2.0",
    "clickhouse-connect",
]

[project.optional-dependencies]
ocr = ["ray[data]", "vllm", "Pillow"]
index = ["ray[data]", "vllm"]
eval = ["jiwer"]

[dependency-groups]
dev = ["pytest", "pyright", "ruff"]

[project.scripts]
mausoleo = "mausoleo.cli:app"
```

### Dependency Groups

- **Core** (just the CLI): lightweight, no GPU dependencies — `uv add mausoleo`
- **ocr**: adds Ray Data and vLLM for running the OCR pipeline — `uv add mausoleo[ocr]`
- **index**: adds Ray Data and vLLM for building the hierarchical index — `uv add mausoleo[index]`
- **eval**: adds evaluation metrics dependencies — `uv add mausoleo[eval]`
- **dev**: pytest, pyright, ruff — `uv sync --group dev`

### Development Workflow

```bash
uv sync --all-extras --all-groups   # install everything
uv run mausoleo --help              # run CLI
uv run pytest                       # run tests
uv run pyright src/                 # type check
```

## 5.4 Server Deployment (Docker Only)

The server (API + ClickHouse) runs exclusively via Docker. Single command to start everything:

```bash
docker compose up -d     # starts clickhouse + mausoleo API server
```

### docker-compose.yml

```yaml
services:
  clickhouse:
    image: clickhouse/clickhouse-server:latest
    volumes:
      - clickhouse_data:/var/lib/clickhouse
    healthcheck:
      test: ["CMD", "clickhouse-client", "--query", "SELECT 1"]
      interval: 5s
      timeout: 3s
      retries: 10

  server:
    build: .
    ports:
      - "8000:8000"
    depends_on:
      clickhouse:
        condition: service_healthy
    environment:
      - CLICKHOUSE_HOST=clickhouse

volumes:
  clickhouse_data:
```

### Dockerfile

```dockerfile
FROM python:3.11-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev
COPY src/ src/
EXPOSE 8000
CMD ["uv", "run", "uvicorn", "mausoleo.server.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
```

ClickHouse is never exposed to the host — only the API server port is published. The CLI talks to the API server, not to ClickHouse directly.

Users never install or manage ClickHouse themselves. `uv add mausoleo` gives them just the CLI. `docker compose up` gives them the full server stack.

## 5.5 Configuration

Use a config file (`mausoleo.toml` or similar) or environment variables:

```toml
[clickhouse]
host = "localhost"
port = 8123
database = "mausoleo"

[server]
host = "0.0.0.0"
port = 8000

[ocr]
model = "Qwen/Qwen2.5-VL-7B"     # or whatever wins the eval
vllm_url = "http://localhost:8001"

[index]
summarization_model = "..."
embedding_model = "BAAI/bge-m3"
vllm_url = "http://localhost:8001"

[data]
source_dir = "/path/to/scraped/pages"
ocr_output_dir = "/path/to/ocr/output"
```

## 5.6 External Dependencies

Document clearly in README:
- Docker (required for running the server — bundles ClickHouse automatically)
- vLLM (user must run for OCR/index building, not needed for search/CLI)
- GPU (only for OCR and index building phases)

## 5.7 Implementation Steps

1. Restructure repo from current layout to target layout
2. Update pyproject.toml with proper dependencies and entry points
3. Move existing code into new structure (scraper, segmentation logic)
4. Write minimal README (what it is, how to install, how to run each phase)
5. Add docker-compose.yml for ClickHouse
6. Ensure `uv sync` works cleanly
7. Ensure `uv run mausoleo --help` works after sync

### Definition of Done

- `uv add mausoleo` works
- `uv run mausoleo --help` shows available commands
- README covers installation and basic usage for each phase
- docker-compose.yml starts ClickHouse
- Pyright passes with strict checking
- Existing tests pass (adapt from current test files)
