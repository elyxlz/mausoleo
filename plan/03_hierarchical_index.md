# Phase 3: Hierarchical Index (ClickHouse)

> **STATUS 2026-07-17: deferred; earlier implementation removed** (git history, c397f0c and earlier). Will be redesigned properly when it starts, on the finished phase-2 corpus. Only standing decisions and open questions live here.

## Standing decisions

- 7-level tree: Paragraph → Article → Day → Month → Year → Decade → Archive (~several M / ~1–1.5M / ~29K / ~960 / 80 / 8 / 1 nodes).
- One ClickHouse `nodes` table: summary + embedding on every node, raw text at leaves only; vector search as escape hatch.
- Deterministic IDs derived from Issue-schema IDs: `1923-03-15_a01_p02` → `1923-03-15_a01` → `1923-03-15` → `1923-03` → `1923` → `1920s` → `archive`.
- Article nodes carry `unit_type` / `headline` / `page_span` straight from the Issue schema.
- Summaries roughly the same size at every level (~200–400 words), entity-rich, written to support a drill-down decision.
- Built bottom-up as a repeatable batch job pinned to a corpus version — re-OCR implies rebuild.
- Input: phase-2 corpus (one Issue-schema JSON per issue).

## Exit criterion

Valid full tree in ClickHouse (parents, children, ordering consistent), every node has summary + embedding, summary spot-checks pass across eras, rebuild from a corpus version is one command.

## Open questions

- `unit_type` handling in summarization — ads/notices aggregated per day rather than summarized one-by-one?
- Summarization + embedding GPU budget (~1.5M article summaries on ripperred) and model choice — measure GPU-s/node with the same discipline as the OCR budget.
- ClickHouse vector/FTS indexing specifics at build time.
