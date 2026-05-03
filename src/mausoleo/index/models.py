from __future__ import annotations

import dataclasses as dc
import datetime as dt
import typing as tp

Level = tp.Literal[
    "paragraph",
    "article",
    "day",
    "month",
    "year",
    "decade",
    "archive",
]

LEVEL_ORDER: dict[str, int] = {
    "paragraph": 0,
    "article": 1,
    "day": 2,
    "month": 3,
    "year": 4,
    "decade": 5,
    "archive": 6,
}


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

    def to_row(self) -> dict[str, tp.Any]:
        return {
            "node_id": self.node_id,
            "level": self.level,
            "parent_id": self.parent_id,
            "position": self.position,
            "date_start": self.date_start,
            "date_end": self.date_end,
            "source": self.source,
            "summary": self.summary,
            "raw_text": self.raw_text,
            "embedding": list(self.embedding),
            "child_count": self.child_count,
        }
