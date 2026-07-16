# Phase 4: Agent Navigation — API + CLI/MCP

> **STATUS 2026-07-17: deferred; earlier implementation removed** (git history). Rebuilt after phase 3. Design sketch, kept lean.

## Goal

Let an LLM agent explore the phase-3 index efficiently: from the archive root down to primary-source paragraphs across 80 years (1880–1959) of Il Messaggero. The agent is the only user — every interface returns structured JSON, no human formatting.

## Interfaces

Three thin layers over one service:

1. **FastAPI server** — owns the ClickHouse connection and query logic.
2. **MCP server** — first-class, not a nice-to-have: in 2026 this is how agents consume tools. Wraps the same operations as native MCP tools.
3. **CLI** (`mausoleo ...`, typer + httpx) — same operations for shell-tool agents and scripting; JSON to stdout.

## Operations

| Operation | API | CLI |
|---|---|---|
| Get node (summary, level, metadata incl. unit_type/headline/pages at article level) | `GET /nodes/{id}` | `mausoleo node <id>` |
| Children, ordered, paginated — the core drill-down | `GET /nodes/{id}/children` | `mausoleo children <id>` |
| Parent | `GET /nodes/{id}/parent` | `mausoleo parent <id>` |
| Full text (leaf raw text; non-leaf reconstructed from descendant paragraphs in order) | `GET /nodes/{id}/text` | `mausoleo text <id>` |
| Archive root (entry point) | `GET /root` | `mausoleo root` |
| Semantic search (level + date-range + unit_type filters) | `POST /search/semantic` | `mausoleo search semantic "<q>" [--level] [--from] [--to] [--limit]` |
| Keyword search | `POST /search/text` | `mausoleo search text "<q>" ...` |
| Index stats (nodes per level, date range, corpus version) | `GET /stats` | `mausoleo stats` |

Hybrid search only if pure semantic or pure keyword proves insufficient in agent testing.

## Agent contract

Tool descriptions ship with the server (MCP tool descriptions / system-prompt snippet): explain the 7-level tree, the strategy (start broad at root, read summaries, drill into relevant branches, use search when the target is known, fetch `text` only at the bottom), and the ID scheme so agents can jump directly to dates (`1922-10`, `1922-10-28`).

The real deliverable of this phase is not endpoints — it is an agent that navigates well. Budget explicit iteration time on tool descriptions and response shapes driven by transcripts of a real agent (Claude) answering test queries.

## Benchmark queries (exit test)

- "Tell me everything about the Pichinon family"
- "How did the collective consciousness of ordinary Romans change during fascism?"
- "Interesting restaurant stories from Trastevere"

## Exit criteria

- All operations functional over the full phase-3 index; MCP + CLI both working against the API.
- An LLM agent answers the benchmark queries, reaching primary-source paragraphs, in a reasonable number of tool calls (target: ≤15 for the entity query).
- Tool descriptions iterated against real agent transcripts at least once.

## Deployment

Server + ClickHouse as one `docker compose up` (schema migrations on startup, ClickHouse never exposed to the host). How this is distributed to anyone else is phase 5's question, not this phase's.
