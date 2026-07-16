# Mausoleo di Roma — Project Roadmap

Updated 2026-07-16 to reflect the OCR-autoresearch refocus. Phase 3+ code was removed from the repo (recoverable from git history) — those phases will be rebuilt on top of the finished OCR corpus. This file is the strategic map; live OCR research state is in `eval/autoresearch/` (program.md, registry.md, log.jsonl, HANDOFF.md).

## Overview

A modern knowledge pipeline for historical newspaper archives. Takes scanned newspaper pages, produces high-quality OCR with correct reading order, builds a recursive hierarchical summary index, and exposes a CLI/API for LLM agents to navigate the knowledge tree efficiently.

Applied to Il Messaggero (Rome), 1880–1959, ~175K pages (~29K issues) stored on endeavour (`/media/sdr/<year>/<Month>/<day>/<N>.jpeg`).

## Phases

| #  | Phase                            | Status |
|----|----------------------------------|--------|
| 01 | OCR eval suite + autoresearch    | ACTIVE — composite_v2 metric, 2 human GT issues + tentative-GT expansion to 4 more eras in progress |
| 02 | Corpus-scale OCR production run  | NEXT — production candidate exists (exp_157: 5.13 GPU-s/page = 5.2-day corpus on 2×3090); needs persistent-engine corpus driver |
| 03 | Hierarchical Index (ClickHouse)  | LATER — rebuild (code removed 2026-07-16, see git history; design in [03_hierarchical_index.md](03_hierarchical_index.md)) |
| 04 | Search & Navigation API + CLI    | LATER — rebuild ([04_search_and_cli.md](04_search_and_cli.md)) |
| 05 | Open Source Packaging            | LATER ([05_packaging.md](05_packaging.md)) |
| 06 | Dissertation                     | LATER — drafts backed up in `~/dissertation_backup_2026-07-16.tar.gz` on ripperred ([06_dissertation.md](06_dissertation.md)) |

## What we learnt (2026-07, supersedes earlier assumptions)

- **Compute budget is the binding constraint, not peak quality.** Corpus target: 6.9–13.9 GPU-s/page steady-state on 2×3090 (1–2 weeks). Any ≥7B full-coverage pass is 5–26× over budget (Qwen3-VL-8B measured at ~136 GPU-s/page). Production quality must come from sub-1B specialized OCR (PaddleOCR-VL-1.6) + cheap layout (YOLO) + CUDA graphs.
- **Eval must charge spam.** composite_v1 rewarded overgeneration; v2 (wCER over all GT, F1 for recall, degenerate-edge fixes) reshuffled the leaderboard: lean 5-source ensemble (0.7776) beats the 8-source (0.7514). Metric changes only via documented reward-hacking audits (`eval_review.md`).
- **Big ensembles are oracles, not products.** `ensemble_30min` (recall ~1.0) is the GT-building/reference tool; `ensemble_prune5` is the v2-optimal reference; neither ships.
- **Structure comes from layout, not the OCR model.** Specialized OCR models (olmOCR, Paddle, GLM, Hunyuan) emit no headings on newspapers; YOLO title-class regions provide headlines/segmentation (exp_152). Long-horizon multi-page parsing (Unlimited-OCR) is blocked — degenerates on this distribution.
- **Experiments are now self-contained scripts** (`experiments/<name>.py <date...>` → `eval/predictions/<name>_<date>.json`) — no framework ceremony; the legacy Ray harness remains only for the verified oracle ensembles.
- **GT strategy:** oracle stack + Fable-subagent reconstruction → `eval/tentative_gt/<date>/` → human review → promotion. In progress for 1895/1925/1935/1952-06-15.

## Key decisions still standing for later phases

- Hierarchy: Paragraph → Article → Day → Month → Year → Decade → Archive (7 levels)
- One ClickHouse node table: summary + embedding + raw text at leaves only
- Summaries are rich text blobs, roughly same size across levels
- Vector search as escape hatch when tree traversal misses
- CLI outputs structured JSON, designed purely for LLM agent consumption
- Newspaper-specific for now; generic system is future work

## Data

- Source: Il Messaggero archive on endeavour, `ssh -p 62420 elio@81.105.49.222`, `/media/sdr/<year>/<MonthName>/<day>/<page>.jpeg`, years 1880–2000 present
- Primary scope 1880–1959 ≈ 175K pages; full holdings ~1.07M pages through 1996
