# Mausoleo di Roma — Project Roadmap

Updated 2026-08-22. Strategic map only; live OCR research state is in `eval/autoresearch/` (program.md, registry.md, LOOP.md, mausoleobench_log.jsonl).

## Product

Turn the complete Il Messaggero archive (Rome, 1880–1959, ~175K scanned pages ≈ 29K issues) into a knowledge system LLM agents can actually use: OCR every page into structured articles (Issue schema), build a recursive hierarchical summary index (paragraph → article → day → month → year → decade → archive), and expose tree navigation + search as agent tools (JSON API / CLI / MCP). The product is the navigable corpus: an agent goes from "what happened to X" to primary-source paragraphs in a handful of tool calls.

## Phases

| # | Phase | Status | Exit criterion |
|---|-------|--------|----------------|
| 01 | [OCR quality research](01_ocr.md) | ACTIVE | In-budget config meets the ship bar in 01_ocr.md on the expanded 6-era GT set |
| 02 | [Corpus production run](02_corpus_run.md) | NOT STARTED — concerns listed, no driver | ≥99% of ~29K issues as valid Issue JSON, versioned on endeavour, probe metrics clean per decade |
| 03 | [Hierarchical index](03_hierarchical_index.md) | LATER — code removed 2026-07-16, redesign at start | Full valid tree in ClickHouse, every node summary + embedding, era spot-checks pass |
| 04 | [Agent navigation](04_search_and_cli.md) | LATER — after 03 | An LLM agent answers the benchmark queries root→paragraph |
| 05 | [Distribution](05_packaging.md) | STUB — form undecided | n/a yet |

Phases 1 and 2 overlap deliberately: a full corpus pass costs ~1 week, so corpus **v0** (best in-budget config) runs early to unblock phase 3, and **v1+** rerun as phase-1 quality improves.

## Quality anchors (composite_v2, 2026-07-16 — authoritative table in program.md)

| Config | v2 avg | GPU cost | Role |
|---|---|---|---|
| `ensemble_prune5` | 0.7776 | oracle-tier | quality reference |
| `ensemble_30min` | 0.7514 | ~600 GPU-s/page | recall oracle for GT building |
| `exp_157` (PaddleOCR-VL-1.6 + YOLO titles) | 0.4284 | **5.13 GPU-s/page = 5.2-day corpus** | production candidate |

The production–oracle gap is recall/segmentation; closing it is phase 1's whole job.

## What we learnt (2026-07, supersedes earlier assumptions)

- **Compute budget is the binding constraint**: cap 250.0 sec/page caller-measured on 2×3090 (target 103.5), raised 5× on 2026-07-22 from the earlier 6.9–13.9 GPU-s/page regime. The raise made 8B column routes affordable; the cheapest good route is still sub-1B OCR + cheap layout.
- **Eval must charge spam** — composite_v2; metric changes only via documented reward-hacking audits.
- **Big ensembles are oracles, not products** — GT building and upper bounds only.
- **Structure comes from layout, not the OCR model** — specialized OCR emits no headings on newspapers; YOLO title regions provide segmentation. Long-horizon multi-page parsing is BLOCKED.
- **Experiments are self-contained scripts**; the legacy Ray harness was removed on 2026-08-22 (oracle numbers survive in the log and `eval/predictions/archive/`).
- **Rerunning the corpus is cheap (~1 week)** — everything downstream must survive a re-OCR with a version bump.

## Key decisions still standing

- 7-level hierarchy; one ClickHouse node table; summary + embedding per node, raw text at leaves; same-sized summaries at every level; vector search as escape hatch.
- All agent-facing output is structured JSON; MCP first-class.
- Newspaper-specific for now; generic multi-archive system is future work.

## Data

- Source: endeavour (`ssh -p 62420 elio@81.105.49.222`), `/media/sdr/<year>/<MonthName>/<day>/<N>.jpeg` — storage only, zero compute.
- Primary scope 1880–1959 ≈ 175K pages; full holdings ~1.07M pages through 1996 (a future corpus version).
- All compute on ripperred (2× RTX 3090, tight disk) — constraints in CLAUDE.md.
