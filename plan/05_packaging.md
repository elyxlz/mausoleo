# Phase 5: Distribution

> **STATUS 2026-07-17: stub.** The old plan (pip package + Docker + ClickHouse bundle) assumed the product was reusable pipeline software. It is not: the value is the **built index over Il Messaggero**, not the code. What ships is undecided.

## Candidate forms (not exclusive; likeliest = hosted service + data release)

- **Hosted service** — the phase-4 API/MCP endpoint over the finished index, run by us.
- **Self-host bundle** — docker compose + downloadable index dump (embeddings make it order tens of GB).
- **Data release** — the raw OCR corpus (Issue JSONs, low GB) as an open dataset.
- **Code release** — this repo as-is; documentation value only (unusable without the corpus + GPUs).

## Open questions (decide before investing anything)

1. **Rights** — can OCR text/summaries of Il Messaggero 1880–1959 be republished? Early decades likely public domain, 1940s–50s unclear. This decides hosted-private vs open-data.
2. **Audience** — historians, agent developers, both?
3. **Corpus versioning** — any public artifact pins a corpus + index version; rebuilds must not silently change published data.

Deliberately not planned: pip-installable pipeline library, generic multi-archive framework. Revisit when phase 4 has a working agent over the real index.
