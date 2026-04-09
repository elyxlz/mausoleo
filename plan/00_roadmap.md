# Mausoleo di Roma — Project Roadmap

## Overview

A modern knowledge pipeline for historical newspaper archives. Takes scanned newspaper pages, produces high-quality OCR with correct reading order, builds a recursive hierarchical summary index in ClickHouse, and exposes a CLI/API for LLM agents to navigate the knowledge tree efficiently.

Applied to Il Messaggero (Rome), 1880–~1945, hundreds of thousands of pages.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Python Package                        │
│                      pip install mausoleo                     │
│                                                               │
│  ┌──────────────┐  ┌──────────────────┐  ┌────────────────┐  │
│  │  OCR Pipeline │  │ Index Builder    │  │  Search/Nav    │  │
│  │  (Ray Data)   │  │ (vLLM + Ray)    │  │  (API Server)  │  │
│  │              │  │                  │  │                │  │
│  │  VLM OCR ────┼──┼─► Recursive     │  │  Tree traversal│  │
│  │  + LLM       │  │   Summarization │  │  Vector search │  │
│  │  cleanup     │  │                  │  │  FTS search    │  │
│  └──────┬───────┘  └────────┬─────────┘  └───────┬────────┘  │
│         │                   │                     │           │
│         └───────────────────┴─────────────────────┘           │
│                             │                                 │
│                      ┌──────▼──────┐                          │
│                      │  ClickHouse  │                          │
│                      └─────────────┘                          │
│                                                               │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │  CLI (typer) — LLM agent tool interface — JSON output    │ │
│  └──────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Phases

| #  | Phase                          | File                        | Depends On |
|----|--------------------------------|-----------------------------|------------|
| 01 | OCR Evaluation Suite           | [01_ocr_eval.md](01_ocr_eval.md)           | —          |
| 02 | OCR Pipeline (Ray Data)        | [02_ocr_pipeline.md](02_ocr_pipeline.md)   | 01         |
| 03 | Hierarchical Index (ClickHouse)| [03_hierarchical_index.md](03_hierarchical_index.md) | 02 |
| 04 | Search & Navigation API + CLI  | [04_search_and_cli.md](04_search_and_cli.md) | 03       |
| 05 | Open Source Packaging          | [05_packaging.md](05_packaging.md)         | 04         |
| 06 | Dissertation                   | [06_dissertation.md](06_dissertation.md)   | 04         |

## Key Decisions (locked in)

- Single pipeline for all eras (1880–~1945), no per-era tuning
- OCR quality bar: LLM does heavy lifting, OCR model is a strong first-pass VLM
- Whole-issue LLM calls (~10 pages/issue) for reading order + cross-page article stitching
- "Article" = LLM-judged semantic unit of information
- Hierarchy: Paragraph → Article → Day → Month → Year → Decade → Archive (7 levels)
- One ClickHouse node table: summary + embedding + raw text at leaves only
- Summaries are rich text blobs (entities/topics woven in, not separate columns)
- Summaries stay roughly same size across all levels
- Vector search as escape hatch when tree traversal misses something
- CLI outputs structured JSON, designed purely for LLM agent consumption
- All Python, single pip install, API-based architecture
- Newspaper-specific for now; generic system is future work
- Scope: Il Messaggero, 1880–~1945

## Data

- Source: Il Messaggero archive, already scraped as JPEGs
- Organization: `data/<year>/<month_name>/<day>/<page_number>.jpeg`
- Estimated scale: ~60 years × ~365 days × ~6 pages/day ≈ 130k pages, ~22k daily issues
- Cut-off at ~WW2 to keep issues under ~10 pages
