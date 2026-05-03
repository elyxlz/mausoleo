# Mausoleo di Roma

Hierarchical knowledge index for historical newspaper archives.

Takes scanned newspaper pages, produces high-quality OCR with correct reading order, builds a recursive hierarchical summary tree in ClickHouse, and exposes a CLI for LLM agents to navigate the knowledge efficiently.

## Architecture

```
Scanned Pages → OCR Pipeline (Ray Data + vLLM) → Structured Text
    → Recursive Summarization (vLLM) → Hierarchical Index (ClickHouse)
        → API Server (FastAPI) → CLI (typer) → LLM Agent
```

### Hierarchy

```
Paragraph → Article → Day → Month → Year → Decade → Archive
```

Each level stores a fixed-size summary. An agent navigates top-down through summaries, drilling into branches of interest, or searches semantically/by keyword across any level.

## Install

```bash
uv add mausoleo              # CLI only
uv add mausoleo[ocr]         # + OCR pipeline dependencies
uv add mausoleo[index]       # + index building dependencies
uv add mausoleo[search]      # + sentence-transformers for query embeddings
```

## How to run

### 1. Start the stack

```bash
docker compose up -d           # ClickHouse + API server on :8000
```

The compose file exposes only the API port; ClickHouse is internal. The
server runs `python -m mausoleo.server.app:create_app` and installs the
schema automatically on startup.

### 2. Load the corpus

The index is populated by walking two directories:

- `eval/transcriptions/*.json` — OCR output (one file per day, with
  articles and paragraphs). These yield level 0 (paragraph) and level 1
  (article) nodes deterministically.
- `eval/summaries/{level}/*.json` — per-level summary nodes produced by
  the summarisation pipeline. Each file is one node, with at minimum
  `node_id`, `level`, `summary`, and optionally `embedding`,
  `parent_id`, `position`, `date_start`, `date_end`, `child_count`.
  Missing files are tolerated; the loader emits structural placeholders so
  the tree is navigable end-to-end before summaries are produced.

```bash
mausoleo load --from 1943-07-01 --to 1943-07-31
```

### 3. Use the CLI

Every command outputs JSON to stdout (no formatting), making the CLI
suitable as a tool for LLM agents.

```bash
mausoleo root                              # archive root node
mausoleo node 1943-07                      # inspect a node
mausoleo children 1943-07 --limit 50       # drill down
mausoleo parent 1943-07-15                 # walk up
mausoleo text 1943-07-15_a01_p00           # raw text (leaf)
mausoleo text 1943-07-15                   # reconstructed (recursively)

mausoleo search semantic "guerra"          # vector search
mausoleo search text "Mussolini"           # FTS via tokenbf_v1 / positionCaseInsensitive
mausoleo search hybrid "Roma"              # RRF combination

mausoleo stats                             # totals per level
```

Server URL precedence: `--server` flag > `MAUSOLEO_SERVER_URL` env var > `http://127.0.0.1:8000`.

### 4. Agent flow

```
mausoleo root                              # → archive
mausoleo children archive                  # → decades
mausoleo children 1940s                    # → years
mausoleo children 1943                     # → months
mausoleo children 1943-07                  # → days
mausoleo children 1943-07-15               # → articles
mausoleo text 1943-07-15_a01               # → reconstructed text
```

## Schema

ClickHouse `nodes` table:

```sql
CREATE TABLE nodes (
    node_id      String,
    level        Enum8('paragraph'=0,'article'=1,'day'=2,'month'=3,
                       'year'=4,'decade'=5,'archive'=6),
    parent_id    String,
    position     UInt32,
    date_start   Date32,
    date_end     Date32,
    source       String DEFAULT 'il_messaggero',
    summary      String,
    raw_text     Nullable(String),
    embedding    Array(Float32),
    child_count  UInt32,
    INDEX summary_idx summary TYPE tokenbf_v1(10240, 3, 0) GRANULARITY 1,
    INDEX embedding_idx embedding TYPE vector_similarity('hnsw','L2Distance',1024) GRANULARITY 1
)
ENGINE = MergeTree
PRIMARY KEY (level, date_start, position);
```

We use `Date32` (not `Date`) so the archive root can span 1880–1945 without overflow.

## Development

```bash
git clone https://github.com/elyxlz/mausoleo_di_roma.git
cd mausoleo_di_roma
uv sync --all-extras
uv run pytest tests/test_end_to_end.py    # requires a running ClickHouse
```

The end-to-end tests boot the FastAPI app via `TestClient` (no separate
uvicorn process), so you only need ClickHouse running on localhost:8123.

## License

MIT
