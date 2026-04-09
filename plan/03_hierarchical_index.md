# Phase 3: Hierarchical Index (ClickHouse)

## Goal

Build the recursive hierarchical summary tree from OCR output and store it in ClickHouse. This is the core intellectual contribution of the project — a navigable, multi-resolution knowledge structure over 60+ years of newspaper data.

## 3.1 Hierarchy Definition

```
Level 0: Paragraph    (leaf node, raw text)
Level 1: Article      (summarizes its paragraphs)
Level 2: Day          (summarizes all articles from one issue)
Level 3: Month        (summarizes all days in a month)
Level 4: Year         (summarizes all months in a year)
Level 5: Decade       (summarizes all years in a decade)
Level 6: Archive      (single root node, summarizes all decades)
```

Summaries are roughly the same size at every level. A decade summary is not longer than an article summary — it's just at a higher level of abstraction.

## 3.2 ClickHouse Schema

### Main Node Table

```sql
CREATE TABLE nodes (
    node_id       String,
    level         Enum8(
                    'paragraph' = 0,
                    'article' = 1,
                    'day' = 2,
                    'month' = 3,
                    'year' = 4,
                    'decade' = 5,
                    'archive' = 6
                  ),
    parent_id     String,
    position      UInt32,
    date_start    Date,
    date_end      Date,
    source        String DEFAULT 'il_messaggero',
    summary       String,
    raw_text      Nullable(String),
    embedding     Array(Float32),
    child_count   UInt32,

    PRIMARY KEY (node_id)
)
ENGINE = MergeTree()
ORDER BY (level, date_start, position);
```

### Design Notes

- `node_id`: deterministic, human-readable (e.g., `1923-03-15_a01_p02`, `1923-03`, `1920s`)
- `parent_id`: points to parent node. Archive node has empty parent.
- `position`: ordering among siblings (e.g., article order within a day, paragraph order within article)
- `date_start` / `date_end`: same for day-level and below; range for month/year/decade/archive
- `raw_text`: only populated for paragraph nodes (level 0), NULL otherwise
- `embedding`: vector embedding of the summary, for semantic search
- `child_count`: number of direct children (useful for the agent to gauge breadth before drilling down)
- `source`: supports future multi-archive expansion

### Vector Index

```sql
ALTER TABLE nodes ADD INDEX embedding_idx embedding TYPE usearch('L2Distance') GRANULARITY 1;
```

### Full-Text Index

```sql
ALTER TABLE nodes ADD INDEX summary_idx summary TYPE tokenbf_v1(10240, 3, 0) GRANULARITY 1;
```

If ClickHouse's FTS is insufficient, evaluate adding a `ngramSearch` or `multiSearchAny` approach. Worst case, maintain a separate lightweight FTS index.

### Key Queries

```sql
-- Tree traversal: get children of a node
SELECT * FROM nodes WHERE parent_id = {node_id} ORDER BY position;

-- Vector search at a specific level
SELECT node_id, summary, L2Distance(embedding, {query_vec}) AS dist
FROM nodes
WHERE level = {target_level}
ORDER BY dist
LIMIT 20;

-- Vector search across all levels
SELECT node_id, level, summary, L2Distance(embedding, {query_vec}) AS dist
FROM nodes
ORDER BY dist
LIMIT 20;

-- Full-text search
SELECT node_id, level, summary
FROM nodes
WHERE hasToken(summary, {term})
ORDER BY level, date_start;

-- Get all leaves (raw text) under a node
-- Recursive: get node's children, then their children, etc. down to paragraphs
-- ClickHouse doesn't do recursive CTEs well, so handle this in application code
-- by walking down levels using parent_id queries
```

## 3.3 Recursive Summarization Pipeline

### Bottom-Up Construction

Build the tree from leaves up:

1. **Paragraphs** (Level 0): directly from OCR output. Raw text = the paragraph text. Summary = the paragraph text itself (or a one-line summary for very long paragraphs).

2. **Articles** (Level 1): for each article, collect its paragraphs. Summarize them into a single rich summary. Include key entities, topics, and the gist of the article.

3. **Days** (Level 2): for each date, collect all article summaries. Summarize into a day overview. What happened in Rome on this day? Key stories, events, themes.

4. **Months** (Level 3): collect all day summaries for the month. Summarize into a month overview. Major events, trends, recurring themes.

5. **Years** (Level 4): collect all month summaries. Summarize into a year overview.

6. **Decades** (Level 5): collect all year summaries. Summarize into a decade overview.

7. **Archive** (Level 6): collect all decade summaries. One root summary of the entire archive.

### Summarization with vLLM

Use vLLM for all summarization inference. Same setup as Phase 2 — vLLM server with Ray Data for orchestration.

**Key prompt design**: each summarization prompt should instruct the LLM to:
- Produce a summary of consistent length (target: ~200-400 words regardless of level)
- Weave in key entities (people, places, organizations)
- Mention notable topics and themes
- At higher levels, capture both major events and the texture of daily life
- Preserve specificity — names, dates, places — rather than vague generalities
- The summary should help an agent decide whether to drill deeper into this node

### Handling Context Limits

- Paragraph → Article: small, fits easily in any context window
- Article → Day: a day might have 15-30 article summaries (~3k-6k words). Fits in any modern LLM.
- Day → Month: 28-31 day summaries (~6k-12k words). Fits comfortably.
- Month → Year: 12 month summaries (~2.4k-4.8k words). Easy.
- Year → Decade: 10 year summaries (~2k-4k words). Easy.
- Decade → Archive: 6-7 decade summaries. Trivial.

No context limit issues at any level. The hardest step is Day (summarizing all articles), and even that is well within limits.

### Embedding Generation

After generating each summary, compute its vector embedding:
- Use a dedicated embedding model (not the summarization LLM)
- Candidates: BGE-M3 (multilingual, good for Italian), multilingual-e5-large, or whatever is best at time of implementation
- Run embedding inference as a separate Ray Data stage after summarization
- Store directly in the embedding column

## 3.4 Node ID Scheme

Deterministic IDs that encode the hierarchy:

| Level     | ID Format                  | Example              |
|-----------|----------------------------|----------------------|
| Paragraph | `{date}_a{nn}_p{nn}`      | `1923-03-15_a01_p02` |
| Article   | `{date}_a{nn}`            | `1923-03-15_a01`     |
| Day       | `{date}`                  | `1923-03-15`         |
| Month     | `{year}-{month}`          | `1923-03`            |
| Year      | `{year}`                  | `1923`               |
| Decade    | `{decade}s`               | `1920s`              |
| Archive   | `archive`                 | `archive`            |

Parent relationships are derivable from IDs, but we store parent_id explicitly for fast queries.

## 3.5 Ray Data Pipeline for Index Building

```
Pipeline stages:
  read_ocr_output(date_range)           # yields structured OCR JSONs
  -> map(create_paragraph_nodes)         # Level 0 nodes
  -> map(create_article_nodes)           # Level 1: summarize paragraphs
  -> [write paragraph + article nodes to ClickHouse]

  # Then aggregate up:
  read_article_nodes(date_range)
  -> group_by(date)
  -> map_batches(summarize_days)         # Level 2: summarize articles per day, GPU
  -> [write day nodes to ClickHouse]

  read_day_nodes(month_range)
  -> group_by(month)
  -> map_batches(summarize_months)       # Level 3, GPU
  -> [write month nodes to ClickHouse]

  # ... continue up to archive level
```

Each level can be built independently once the level below is complete. This is naturally sequential (can't summarize months before days exist), but within each level, all nodes can be processed in parallel.

## 3.6 Implementation Steps

1. Set up ClickHouse (local Docker instance for development)
2. Create the schema (nodes table, indexes)
3. Write node creation logic for Level 0 (paragraphs) and Level 1 (articles) from OCR output
4. Write the summarization prompt template
5. Implement Level 1 summarization (article summaries from paragraphs) using vLLM
6. Implement Level 2 summarization (day summaries from articles)
7. Test on a small subset (~1 month of data) — verify tree structure, summary quality
8. Implement Levels 3–6 summarization
9. Implement embedding generation and storage
10. Run on full OCR output
11. Verify tree integrity: every non-root node has a valid parent, every non-leaf has children, ordering is consistent
12. Spot-check summary quality at each level across eras

### Definition of Done

- ClickHouse populated with the full tree (~millions of paragraph nodes, ~hundreds of thousands of article nodes, ~22k day nodes, ~780 month nodes, ~66 year nodes, ~7 decade nodes, 1 archive node)
- Every node has a summary and embedding
- Tree structure is valid and navigable
- Summary quality is acceptable at all levels (spot-checked)
