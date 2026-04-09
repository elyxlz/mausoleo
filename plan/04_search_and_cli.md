# Phase 4: Search & Navigation API + CLI

## Goal

Build the API server and CLI that let an LLM agent explore the hierarchical knowledge index. The agent should be able to efficiently navigate from the archive root down to specific primary sources across 60+ years of newspaper data.

## 4.1 API Server

### Tech Stack

- FastAPI (already a dependency)
- Async ClickHouse client (clickhouse-connect or asynch)
- Embedding model loaded in-process for query embedding (or call out to a separate service)

### Deployment: Docker Only

The server runs exclusively via Docker. A single `docker compose up` starts both the API server and ClickHouse together. No manual ClickHouse setup, no systemd — fully self-contained.

```
docker compose up        # starts clickhouse + mausoleo server
docker compose down      # stops everything
```

On startup, the server waits for ClickHouse to be healthy (Docker healthcheck + retry loop), then runs schema migrations automatically.

### Endpoints

#### Tree Traversal

**`GET /nodes/{node_id}`**
Returns a single node with its summary, level, metadata.

```json
{
  "node_id": "1923-03",
  "level": "month",
  "parent_id": "1923",
  "date_start": "1923-03-01",
  "date_end": "1923-03-31",
  "summary": "...",
  "child_count": 31,
  "source": "il_messaggero"
}
```

**`GET /nodes/{node_id}/children`**
Returns all direct children of a node, ordered by position. This is the core "drill down" operation.

Optional params:
- `offset`, `limit` for pagination (months with 30+ days, days with many articles)

```json
{
  "parent_id": "1923-03",
  "children": [
    {"node_id": "1923-03-01", "level": "day", "position": 1, "summary": "..."},
    {"node_id": "1923-03-02", "level": "day", "position": 2, "summary": "..."},
    ...
  ]
}
```

**`GET /nodes/{node_id}/parent`**
Navigate up the tree.

**`GET /nodes/{node_id}/text`**
For leaf nodes (paragraphs): return raw text.
For non-leaf nodes: reconstruct full text by fetching all descendant paragraphs in order.

#### Search

**`POST /search/semantic`**
Vector similarity search across the index.

```json
{
  "query": "public reaction to March on Rome",
  "level": "article",        // optional: filter by level
  "date_start": "1922-10-01", // optional: date range filter
  "date_end": "1922-11-30",
  "limit": 20
}
```

Returns nodes ranked by similarity with their summaries.

**`POST /search/text`**
Full-text / keyword search.

```json
{
  "query": "Pichinon",
  "level": null,              // search all levels
  "date_start": null,
  "date_end": null,
  "limit": 20
}
```

**`POST /search/hybrid`**
Combined semantic + text search with configurable weighting. Optional — implement if pure semantic or pure text search proves insufficient.

#### Utility

**`GET /root`**
Returns the archive root node. Entry point for top-down traversal.

**`GET /stats`**
Returns index statistics: total nodes per level, date range, source archives.

## 4.2 CLI (Agent Tool Interface)

### Tech Stack

- typer for CLI framework
- httpx for API calls
- All output as structured JSON (the CLI user is an LLM agent)

### Commands

```bash
# Tree navigation
mausoleo node <node_id>                    # get node details
mausoleo children <node_id>                # list children
mausoleo parent <node_id>                  # go up
mausoleo text <node_id>                    # get full text (leaf or reconstructed)
mausoleo root                              # get archive root

# Search
mausoleo search semantic "<query>" [--level <level>] [--from <date>] [--to <date>] [--limit N]
mausoleo search text "<query>" [--level <level>] [--from <date>] [--to <date>] [--limit N]

# Utility
mausoleo stats                             # index statistics
```

### Output Format

Every command outputs JSON to stdout. No human-friendly formatting, no colors, no tables. Pure structured data for the agent to parse.

```bash
$ mausoleo children 1923-03-15
{
  "parent_id": "1923-03-15",
  "children": [
    {"node_id": "1923-03-15_a01", "level": "article", "position": 1, "summary": "..."},
    ...
  ]
}
```

### Configuration

```bash
mausoleo --server http://localhost:8000    # API server URL
```

Or via environment variable `MAUSOLEO_SERVER_URL`.

## 4.3 Agent Integration

### Tool Descriptions for LLM Agents

The CLI commands should be described as tools that an LLM agent can use. Write clear tool descriptions that explain what each command does and when to use it.

Example MCP tool description or system prompt snippet:

```
You have access to a historical newspaper knowledge index covering Il Messaggero (Rome) from 1880 to ~1945. The index is organized as a hierarchical tree:

Paragraph → Article → Day → Month → Year → Decade → Archive

You can navigate this tree and search it using these tools:
- `mausoleo root` — start here. Returns the archive root with a high-level summary.
- `mausoleo children <node_id>` — drill down. See all items at the next level of detail.
- `mausoleo node <node_id>` — inspect a specific node's summary.
- `mausoleo text <node_id>` — read the full original text of an article or paragraph.
- `mausoleo search semantic "<query>"` — find relevant nodes by meaning.
- `mausoleo search text "<query>"` — find nodes by keyword.

Strategy: start broad (root → decades → years), read summaries to identify relevant branches, drill down to articles and paragraphs for primary sources. Use search when you know what you're looking for.
```

### Potential MCP Server

Consider wrapping the API as an MCP (Model Context Protocol) server so Claude and other agents can use it natively as tools without CLI invocation. This is a nice-to-have, not essential.

## 4.4 Implementation Steps

1. Set up FastAPI server skeleton with ClickHouse connection
2. Implement `GET /nodes/{node_id}` and `GET /nodes/{node_id}/children`
3. Implement `GET /nodes/{node_id}/text` (with recursive descendant fetching for non-leaves)
4. Implement `POST /search/semantic` (embed query, ANN search in ClickHouse)
5. Implement `POST /search/text`
6. Implement remaining endpoints (parent, root, stats)
7. Build CLI with typer wrapping all endpoints
8. Write agent tool descriptions
9. Test with an actual LLM agent (Claude via CLI tools) on real queries
10. Iterate on tool descriptions and output format based on how well the agent navigates

### Definition of Done

- API server running, all endpoints functional
- CLI installed via pip, all commands working
- An LLM agent can successfully answer the example queries from our brainstorm:
  - "Tell me everything about the Pichinon family"
  - "How did the collective consciousness of ordinary Romans change during fascism?"
  - "Interesting restaurant stories from Trastevere"
- Agent can navigate from root to primary source paragraphs in a reasonable number of tool calls
