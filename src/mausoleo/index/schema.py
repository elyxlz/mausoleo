from __future__ import annotations

DROP_NODES_TABLE = "DROP TABLE IF EXISTS nodes"

CREATE_NODES_TABLE = """
CREATE TABLE IF NOT EXISTS nodes (
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
    date_start    Date32,
    date_end      Date32,
    source        String DEFAULT 'il_messaggero',
    summary       String,
    raw_text      Nullable(String),
    embedding     Array(Float32),
    child_count   UInt32,

    INDEX summary_idx summary TYPE tokenbf_v1(10240, 3, 0) GRANULARITY 1
)
ENGINE = MergeTree()
ORDER BY (level, date_start, position)
PRIMARY KEY (level, date_start, position)
SETTINGS index_granularity = 8192
"""

# Vector index added separately; uses ClickHouse's vector_similarity HNSW
# index. Three required args: method, distance, dimensionality. We tolerate
# failure if the experimental setting isn't enabled — the L2Distance() function
# still works without it.
CREATE_EMBEDDING_INDEX = """
ALTER TABLE nodes ADD INDEX IF NOT EXISTS embedding_idx embedding
TYPE vector_similarity('hnsw', 'L2Distance', 1024) GRANULARITY 1
"""

# Helper SQL for the application layer.
COUNT_NODES = "SELECT count() FROM nodes"

STATS_BY_LEVEL = """
SELECT level, count() AS n, min(date_start) AS d_min, max(date_end) AS d_max
FROM nodes
GROUP BY level
ORDER BY level
"""


def all_setup_statements() -> list[str]:
    """SQL statements to run, in order, to set up the schema."""
    return [CREATE_NODES_TABLE, CREATE_EMBEDDING_INDEX]
