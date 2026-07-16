# Mausoleo di Roma — Project Roadmap

Updated 2026-07-17. This file is the strategic map. Live OCR research state lives in `eval/autoresearch/` (program.md, registry.md, log.jsonl); this file never duplicates numbers found there beyond the anchors below.

## Product

Turn the complete Il Messaggero archive (Rome, 1880–1959, ~175K scanned pages ≈ 29K issues on endeavour) into a knowledge system LLM agents can actually use: OCR every page into structured articles (Issue schema), build a recursive hierarchical summary index (paragraph → article → day → month → year → decade → archive), and expose tree navigation + search as agent tools (structured-JSON CLI / API / MCP). The finished product is the navigable corpus: an agent goes from "what happened to X" to primary-source paragraphs in a handful of tool calls. Not a dissertation, not a demo — a working archive product.

## Phases

| # | Phase | Status | Exit criteria |
|---|-------|--------|---------------|
| 01 | [OCR quality research](01_ocr.md) (autoresearch loop) | ACTIVE | Production config within budget (≤13.9 GPU-s/page) meets the ship bar in 01_ocr.md on the expanded GT set (6 eras) |
| 02 | [Corpus-scale production run](02_corpus_run.md) | DESIGNED — driver not built | ≥99% of ~29K issues emit valid Issue JSON; probe metrics stable per decade; versioned corpus + run manifest on endeavour |
| 03 | [Hierarchical index](03_hierarchical_index.md) | LATER — code removed 2026-07-16 (git history), rebuild on finished corpus | Full 7-level tree in ClickHouse, every node has summary + embedding, tree valid, spot-checks pass per era |
| 04 | [Agent navigation: API + CLI/MCP](04_search_and_cli.md) | LATER — rebuild after 03 | An LLM agent answers the benchmark queries root→paragraph in a reasonable number of tool calls |
| 05 | [Distribution](05_packaging.md) | STUB — form of the product undecided | Decided + shipped distribution (hosted service and/or self-host bundle) |

Phase 1 and phase 2 overlap deliberately: a full corpus pass costs ~1 week of GPU, so the corpus run is a repeatable batch job, not a one-shot. Corpus **v0** (best in-budget config, unblocks phase 3 development on real data) can run before the phase-1 ship bar; corpus **v1** (the one phase 3 ships on) waits for it. See 02_corpus_run.md "When to trigger".

## Quality anchors (composite_v2, 2026-07-16 — authoritative table in program.md)

| Config | v2 avg | GPU cost | Role |
|---|---|---|---|
| `ensemble_prune5` | 0.7776 | oracle-tier (~18 min/issue) | v2 leader, quality reference |
| `ensemble_30min` | 0.7514 | ~600 GPU-s/page | recall oracle (recall 1.0/0.98) for GT building |
| `exp_157` (PaddleOCR-VL-1.6 + YOLO titles + squeeze) | 0.4284 | **5.13 GPU-s/page = 5.2-day corpus** | production candidate |

The production–oracle gap (0.43 vs 0.78) is dominated by recall/segmentation (exp_157 recall 0.36–0.49), not character accuracy of matched text. Closing it is phase 1's whole job.

## What we learnt (2026-07, supersedes earlier assumptions)

- **Compute budget is the binding constraint, not peak quality.** Corpus target: 6.9–13.9 GPU-s/page steady-state on 2×3090 (1–2 weeks). Any ≥7B full-coverage pass is 5–26× over budget (Qwen3-VL-8B measured ~136 GPU-s/page). Production quality must come from sub-1B specialized OCR (PaddleOCR-VL-1.6, ~5–10 GPU-s/page) + cheap layout (YOLO) + CUDA graphs.
- **Eval must charge spam.** composite_v1 rewarded overgeneration; v2 (wCER over all GT, F1 for recall, degenerate-edge fixes) reshuffled the leaderboard. Metric changes only via documented reward-hacking audits (`eval_review.md`).
- **Big ensembles are oracles, not products.** They set the quality upper bound and build GT; neither ships.
- **Structure comes from layout, not the OCR model.** Specialized OCR models (olmOCR, Paddle, GLM, Hunyuan) emit no headings on newspapers; YOLO title-class regions provide headlines/segmentation (exp_152). Long-horizon multi-page parsing (Unlimited-OCR) is BLOCKED — degenerates on this distribution.
- **Experiments are self-contained scripts** (`experiments/<name>.py <date...>` → `eval/predictions/<name>_<date>.json`); the legacy Ray harness remains only for the oracle ensembles + exp_157.
- **GT strategy:** oracle stack + subagent reconstruction → `eval/tentative_gt/<date>/` → human review → promotion. 2 issues human-verified (1885-06-15, 1910-06-15); 4 era-diverse drafts in flight (1895/1925/1935/1952-06-15).
- **Rerunning the corpus is cheap** (~1 week). Design everything downstream (index IDs, storage layout) to survive a corpus re-OCR with a version bump.

## Key decisions still standing for later phases

- Hierarchy: Paragraph → Article → Day → Month → Year → Decade → Archive (7 levels)
- One ClickHouse node table: summary + embedding at every node, raw text at leaves only
- Summaries are rich text blobs, roughly the same size at every level
- Vector search as escape hatch when tree traversal misses
- All agent-facing output is structured JSON — designed for LLM consumption, not humans; MCP is a first-class interface, not a nice-to-have
- Newspaper-specific for now; generic multi-archive system is future work

## Data

- Source: Il Messaggero archive on endeavour (`ssh -p 62420 elio@81.105.49.222`), `/media/sdr/<year>/<MonthName>/<day>/<N>.jpeg`; storage only, zero compute
- Primary scope 1880–1959 ≈ 175K pages ≈ 29K issues; full holdings ~1.07M pages through 1996 (later decades are a future corpus version)
- All compute on ripperred (2× RTX 3090, tight disk) — see CLAUDE.md constraints
