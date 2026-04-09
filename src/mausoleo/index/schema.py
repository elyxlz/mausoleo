from __future__ import annotations

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
    date_start    Date,
    date_end      Date,
    source        String DEFAULT 'il_messaggero',
    summary       String,
    raw_text      Nullable(String),
    embedding     Array(Float32),
    child_count   UInt32
)
ENGINE = MergeTree()
ORDER BY (level, date_start, position)
PRIMARY KEY (node_id);
"""

CREATE_EMBEDDING_INDEX = """
ALTER TABLE nodes ADD INDEX IF NOT EXISTS embedding_idx embedding TYPE usearch('L2Distance') GRANULARITY 1;
"""

CREATE_FTS_INDEX = """
ALTER TABLE nodes ADD INDEX IF NOT EXISTS summary_idx summary TYPE tokenbf_v1(10240, 3, 0) GRANULARITY 1;
"""
