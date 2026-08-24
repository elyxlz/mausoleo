# Mausoleo di Roma — Roadmap

Updated 2026-08-22. Strategic map only; live OCR research state is in `eval/autoresearch/` (`program.md`, `registry.md`, `LOOP.md`, `mausoleobench_log.jsonl`).

## Product

Turn the complete Il Messaggero archive (Rome, 1880–1959, ~175K scanned pages ≈ 29K issues) into a knowledge system LLM agents can actually use: OCR every page into structured articles (Issue schema), build a recursive hierarchical summary index (paragraph → article → day → month → year → decade → archive), and expose tree navigation + search as agent tools (JSON API / CLI / MCP). The product is the navigable corpus: an agent goes from "what happened to X" to primary-source paragraphs in a handful of tool calls.

## Phases

| # | Phase | Status | Exit criterion |
|---|-------|--------|----------------|
| 1 | OCR quality research | ACTIVE | In-budget config meets the ship bar below on the 6-era GT set |
| 2 | Corpus production run | NOT STARTED — concerns listed, no driver | ≥99% of ~29K issues as valid Issue JSON, versioned on endeavour, probe metrics clean per decade |
| 3 | Hierarchical index | LATER — code removed 2026-07-16, redesign at start | Full valid tree in ClickHouse, every node summary + embedding, era spot-checks pass |
| 4 | Agent navigation | LATER — after phase 3 | An LLM agent answers the benchmark queries root→paragraph |
| 5 | Distribution | STUB — form undecided | n/a yet |

Phases 1 and 2 overlap deliberately: a full corpus pass costs ~1 week, so corpus **v0** (best in-budget config) runs early to unblock phase 3, and **v1+** rerun as phase-1 quality improves.

## Quality anchors (MausoleoBench, from `mausoleobench_log.jsonl`)

| Config | MausoleoBench | sec/page | Role |
|---|---|---|---|
| `ensemble_30min` | 0.5941 | ~600 | recall oracle, GT building — over cap |
| `ensemble_prune5` | 0.5622 | ~400 | pruned oracle reference — over cap |
| `exp_017` Qwen3-VL-8B column route | **0.4275** | 181.98 | current record, in budget |
| `exp_009` article-level PaddleOCR-VL + fill guard | 0.4071 | 8.66 | cheapest good route |

The production–oracle gap is **recall/segmentation**, not the accuracy of matched text. Closing it is phase 1's whole job. (Pre-2026-07-22 anchors were scored in the retired composite_v2 metric under a 6.9–13.9 GPU-s/page budget — git history.)

## Phase 1 — OCR quality research (ACTIVE)

The autoresearch loop hillclimbing MausoleoBench under the budget cap (200.0 sec/page, caller-measured). Experiments are self-contained scripts (`experiments/README.md`); the cycle runs via `scripts/research.py` (eval + audit + holdout + probe), one variable per experiment, everything logged. Metrics and GT are never touched to improve a score.

**GT: 6 issues, final size per Elio (2026-07-17)** — 1885/1910-06-15 human-verified from the start, 1895/1925/1935/1952-06-15 promoted to full GT on 2026-07-21 via `scripts/review_server.py`. The 1925 issue is *Il Meridiano* (the publisher's Monday paper) — accepted. No issue-level held-out set (per Elio); anti-overfit protection is article-level (even/odd holdout halves) plus the GT-free 1943 probes.

**Ship bar for corpus v1 (proposed — needs Elio's sign-off):** one in-budget config with MausoleoBench ≥ 0.60 avg, ≥ 0.50 on every issue (no era collapse), recall ≥ 0.70 avg, ≤ 200.0 sec/page, no probe degradation on the 1943 set. Rationale: phase 3 can summarize noisy text but never recovers articles OCR missed — recall losses are permanent. Corpus v0 may run before the bar. If the bar proves unreachable within budget, that's a product decision (relax bar vs budget), not something to paper over.

**Open bets** (statuses in `registry.md`): F3 layout/reading-order (PP-DocLayoutV3 regions, top bet for recall); F1 sub-1B refinements; F7 segmentation adapters (embedding-similarity grouping); F4 oracle-only precision filtering. **BLOCKED**: F2 long-horizon cross-page parsing (top failure category, waiting on a new model release), HunyuanOCR, GLM-OCR, olmOCR.

## Phase 2 — Corpus-scale production run (NOT STARTED)

Runs the best in-budget config over all ~29K issues. Cheap enough to rerun, so plan for corpus versions.

Known concerns (solve when building, not before): persistent engine (models loaded once per GPU worker, steady-state batching, same code path as eval runs — never a production fork); resumability across ~29K issues (one output file per issue is the checkpoint, restart = manifest diff, per-issue failure isolation + quarantine list); staging (corpus JPEGs don't fit on ripperred — bounded prefetch spool from endeavour, delete after commit, versioned outputs pushed back); GT-free quality monitoring during the run (per-issue probe metrics vs rolling per-decade baselines, GT issues flowing through as sentinels).

**Trigger:** v0 as soon as the GT set shows no era collapse for the production config and a pilot slice (~a month per decade) survives a mid-run kill/resume — do not wait for the ship bar. v1 (what phase 3 ships on) waits for the bar.

**Open questions:** partial-issue policy for corrupt/missing pages; where corpus copies live besides endeavour (single-copy risk).

## Phase 3 — Hierarchical index, ClickHouse (LATER)

Standing decisions: 7-level tree Paragraph → Article → Day → Month → Year → Decade → Archive (~several M / ~1–1.5M / ~29K / ~960 / 80 / 8 / 1 nodes); one ClickHouse `nodes` table with summary + embedding on every node, raw text at leaves only, vector search as escape hatch; deterministic IDs derived from Issue-schema IDs (`1923-03-15_a01_p02` → … → `archive`); article nodes carry `unit_type` / `headline` / `page_span` straight from the Issue schema; summaries roughly the same size at every level (~200–400 words), entity-rich, written to support a drill-down decision; built bottom-up as a repeatable batch job pinned to a corpus version — re-OCR implies rebuild.

**Exit:** valid full tree (parents, children, ordering consistent), every node has summary + embedding, spot-checks pass across eras, rebuild from a corpus version is one command.

**Open questions:** `unit_type` handling in summarization (ads/notices aggregated per day?); summarization + embedding GPU budget (~1.5M article summaries) and model choice, measured with the same discipline as the OCR budget; ClickHouse vector/FTS indexing specifics.

## Phase 4 — Agent navigation, API + CLI/MCP (LATER)

Standing decisions: agent-first (the only user is an LLM agent; every interface returns structured JSON, no human formatting); MCP first-class alongside a typer CLI, both thin layers over one FastAPI + ClickHouse service; core operations node / children / parent / text / root / semantic search / keyword search / stats, with level, date-range and unit_type filters; tool descriptions ship with the server and get iterated against real agent transcripts; deployment is a single `docker compose up` (server + ClickHouse, DB never exposed).

**Exit:** an LLM agent answers the benchmark queries — "tell me everything about the Pichinon family", "how did the collective consciousness of ordinary Romans change during fascism?", "interesting restaurant stories from Trastevere" — reaching primary-source paragraphs in a reasonable number of tool calls.

**Open questions:** is hybrid search needed, or do semantic + keyword suffice; response shapes and pagination — decide from real agent transcripts, not upfront.

## Phase 5 — Distribution (STUB)

The value is the **built index over Il Messaggero**, not the code. Candidate forms (likeliest = hosted service + data release): hosted phase-4 API/MCP endpoint; self-host bundle (docker compose + index dump, tens of GB with embeddings); data release of the raw OCR corpus (Issue JSONs, low GB); code release (documentation value only, unusable without the corpus + GPUs).

**Decide before investing anything:** rights (can OCR text/summaries of Il Messaggero 1880–1959 be republished? early decades likely public domain, 1940s–50s unclear — this decides hosted-private vs open-data); audience (historians, agent developers, both?); corpus versioning (any public artifact pins a corpus + index version; rebuilds must not silently change published data).

Deliberately not planned: pip-installable pipeline library, generic multi-archive framework.

## Key decisions still standing

- All agent-facing output is structured JSON; MCP first-class.
- Newspaper-specific for now; a generic multi-archive system is future work.
- Big ensembles are oracles, not products — GT building and upper bounds only.
- Structure comes from layout, not the OCR model — specialized OCR emits no headings on newspapers.
- Rerunning the corpus is cheap (~1 week) — everything downstream must survive a re-OCR with a version bump.

## Data

- Source: endeavour (`ssh -p 62420 elio@81.105.49.222`), `/media/sdr/<year>/<MonthName>/<day>/<N>.jpeg` — storage only, zero compute.
- Primary scope 1880–1959 ≈ 175K pages; full holdings ~1.07M pages through 1996 (a future corpus version).
- All compute on ripperred (2× RTX 3090, tight disk) — constraints in `CLAUDE.md`.
