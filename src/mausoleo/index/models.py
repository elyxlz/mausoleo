from __future__ import annotations

import dataclasses as dc
import datetime as dt
import typing as tp

Level = tp.Literal["paragraph", "article", "day", "month", "year", "decade", "archive"]


@dc.dataclass(frozen=True)
class Node:
    node_id: str
    level: Level
    parent_id: str
    position: int
    date_start: dt.date
    date_end: dt.date
    source: str
    summary: str
    raw_text: str | None
    embedding: list[float]
    child_count: int
