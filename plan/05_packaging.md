# Phase 5: Distribution

> **STATUS 2026-07-17: stub.** The old plan (pip package `mausoleo` + Docker + ClickHouse bundle, generic repo restructure) assumed the product was reusable pipeline software. It is not: the OCR research loop is corpus- and hardware-specific, and the value is the **built index over Il Messaggero**, not the code. What "distribution" means is genuinely undecided — this file lists the honest options and blockers, nothing more.

## What could actually ship

| Form | What it is | Notes |
|---|---|---|
| Hosted service | The phase-4 API/MCP endpoint over the finished index, run by us | Most product-like; smallest distribution surface; agents connect via MCP URL |
| Self-host bundle | `docker compose` (ClickHouse + server) + a downloadable index dump (node table incl. embeddings) | Dump size is nontrivial (millions of nodes × embedding vectors — order tens of GB); needs a versioned artifact per corpus version |
| Data release | The raw OCR corpus (`ocr_corpus/v<N>/` Issue JSONs) as an open dataset | Low GB, easy to host; useful to others even without our index |
| Code release | This repo as-is (research loop + eval harness) | Cheap to do, but nobody can run it without the corpus and 2×3090s; documentation-only value |

These are not exclusive; likeliest shape is hosted service + data release.

## Blockers / open questions (decide before investing anything here)

1. **Rights.** Can OCR text and summaries of Il Messaggero 1880–1959 be republished? Italian copyright on newspaper content likely expires for the early decades but not necessarily the 1940s–50s. This single question decides between "hosted/private" and "open data" — resolve before building any distribution machinery.
2. **Audience.** Who is this for besides us — historians, agent developers, both? Hosted MCP targets agent developers; a data release targets researchers.
3. **Corpus versioning.** Any public artifact must be pinned to a corpus version (phase 2) and index build; re-OCR + rebuild must not silently change published data.
4. **Name.** `mausoleo` still fine; decide only when something actually ships.

## What is deliberately NOT planned

- No pip-installable pipeline package: the OCR pipeline is not a reusable library and pretending otherwise costs restructure work with no user.
- No generic multi-archive framework: explicitly future work (roadmap decision).

Revisit this file when phase 4 has a working agent over the real index.
