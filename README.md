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
uv add mausoleo[all]         # everything
```

## Server

```bash
docker compose up -d          # starts API server + ClickHouse
```

## Usage

```bash
mausoleo root                              # archive root node
mausoleo children <node_id>                # drill down
mausoleo node <node_id>                    # inspect a node
mausoleo text <node_id>                    # raw text
mausoleo search "<query>" --mode semantic  # semantic search
mausoleo search "<query>" --mode text      # keyword search
mausoleo stats                             # index statistics
```

## Development

```bash
git clone https://github.com/elyxlz/mausoleo_di_roma.git
cd mausoleo_di_roma
uv sync --all-extras
```

## License

MIT
