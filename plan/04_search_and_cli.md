# Phase 4: Agent Navigation — API + CLI/MCP

> **STATUS 2026-07-17: deferred; earlier implementation removed** (git history). Redesigned when it starts, after phase 3. Only standing decisions and open questions live here.

## Standing decisions

- Agent-first: the only user is an LLM agent; every interface returns structured JSON, no human formatting.
- MCP is first-class alongside a typer CLI — both thin layers over one FastAPI + ClickHouse service.
- Core operations: node / children (the drill-down) / parent / text (leaf raw or reconstructed) / root / semantic search / keyword search / stats, with level, date-range, and unit_type filters.
- Tool descriptions ship with the server and get iterated against real agent transcripts — the deliverable is an agent that navigates well, not endpoints.
- Deployment: single `docker compose up` (server + ClickHouse, DB never exposed). Distribution beyond us is phase 5's question.

## Exit criterion

An LLM agent answers the benchmark queries — "tell me everything about the Pichinon family", "how did the collective consciousness of ordinary Romans change during fascism?", "interesting restaurant stories from Trastevere" — reaching primary-source paragraphs in a reasonable number of tool calls.

## Open questions

- Is hybrid search needed, or do semantic + keyword suffice for agent use?
- Response shapes and pagination — decide from real agent transcripts, not upfront.
