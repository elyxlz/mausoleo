# Phase 3: Hierarchical Index (ClickHouse)

> **STATUS 2026-07-17: deferred; earlier implementation removed from the repo** (git history, c397f0c and earlier). Rebuilt from scratch on the finished corpus (phase 2 output). This is a design sketch to be revalidated then — kept lean on purpose.

## Goal

Build the recursive hierarchical summary tree over the OCR corpus and store it in ClickHouse: a multi-resolution structure an LLM agent navigates top-down, reading same-sized summaries at every level and drilling into raw paragraphs only at the leaves.

## Input

Phase-2 corpus: `ocr_corpus/v<N>/<year>/<date>.json`, Issue schema —

```json
{
  "date": "1885-06-15", "source": "il_messaggero", "page_count": 4,
  "articles": [{
    "id": "1885-06-15_a00",
    "unit_type": "article | advertisement | obituary | notice | editorial | other",
    "headline": "text or null",
    "paragraphs": [{"id": "1885-06-15_a00_p00", "text": "..."}],
    "page_span": [1],
    "position_in_issue": 0
  }]
}
```

The index stores the corpus version it was built from; a corpus re-OCR (expected — reruns are ~1 week) triggers an index rebuild, so the builder must be a repeatable batch job, not a one-off migration.

## Hierarchy

```
Level 0: Paragraph   (leaf, raw text)          ~several M nodes
Level 1: Article     (Issue-schema article)    ~1–1.5M nodes
Level 2: Day         (one issue)               ~29K nodes
Level 3: Month                                 ~960 nodes
Level 4: Year                                  80 nodes
Level 5: Decade                                8 nodes
Level 6: Archive                               1 node
```

Summaries are roughly the same size at every level (~200–400 words): a decade summary is not longer than an article summary, just more abstract.

## Schema

```sql
CREATE TABLE nodes (
    node_id       String,
    level         Enum8('paragraph'=0,'article'=1,'day'=2,'month'=3,'year'=4,'decade'=5,'archive'=6),
    parent_id     String,
    position      UInt32,
    date_start    Date,
    date_end      Date,
    source        String DEFAULT 'il_messaggero',
    unit_type     LowCardinality(String) DEFAULT '',
    headline      Nullable(String),
    pages         Array(UInt8),
    summary       String,
    raw_text      Nullable(String),
    embedding     Array(Float32),
    child_count   UInt32,
    corpus_version LowCardinality(String),
    PRIMARY KEY (node_id)
)
ENGINE = MergeTree()
ORDER BY (level, date_start, position);
```

- `node_id` deterministic, derived from Issue-schema IDs: `1923-03-15_a01_p02` / `1923-03-15_a01` / `1923-03-15` / `1923-03` / `1923` / `1920s` / `archive`.
- `unit_type`, `headline`, `pages` populated at article level straight from the Issue schema (`page_span` → `pages`); empty/null above and below.
- `raw_text` only at paragraph nodes; `position` = `position_in_issue` at article level, paragraph order at leaf level.
- `child_count` lets the agent gauge breadth before drilling.
- Vector index (usearch/L2) on `embedding`; token-bloom FTS index on `summary`. Recursive descent handled in application code (walk `parent_id` level by level).

## Construction

Bottom-up, level by level; each level is embarrassingly parallel once the level below exists. All inference on ripperred via vLLM (text-only model — candidate chosen at build time), embeddings from a dedicated multilingual model (BGE-M3 class) as a second pass.

Prompt requirements per summary: consistent length, weave in entities (people, places, organizations), preserve specificity over generality, capture both major events and the texture of daily life at higher levels, and be useful for a drill-down decision. Context fits comfortably at every level (worst case day-level: ~15–50 article summaries).

Open design points, deliberately unresolved until rebuild:
- `unit_type` handling in summarization — ads/notices probably summarized in aggregate per day rather than one summary each (they are ~a third of 1885 GT articles; per-node summaries would be noise).
- Summarization GPU budget: ~1.5M article summaries + 30K higher-level summaries needs its own GPU-s/node measurement against a wall-clock target, same discipline as the OCR budget.
- Headline-aware summaries at article level (headline is free signal; hCER quality from phase 1 matters here).

## Exit criteria

- ClickHouse populated for the full corpus version: valid tree (every non-root has a parent, every non-leaf has children, positions consistent), every node has summary + embedding.
- Node counts within expected ranges per level (table above).
- Summary spot-checks pass at every level across eras (1880s, 1910s, 1930s, 1950s), including known-noisy OCR issues.
- Rebuild-from-scratch is a single command against a corpus version.
