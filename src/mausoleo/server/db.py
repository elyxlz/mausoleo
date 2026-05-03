"""ClickHouse client wrapper for the API server.

We use ``clickhouse_connect``'s synchronous client wrapped behind a small
adapter. clickhouse-connect ships an async client too, but its API is
slightly different and not needed for our throughput; the sync calls run in
FastAPI's worker thread pool.
"""
from __future__ import annotations

import dataclasses as dc
import os
import typing as tp

import clickhouse_connect
from clickhouse_connect.driver.client import Client


@dc.dataclass(frozen=True)
class DbConfig:
    host: str = "127.0.0.1"
    port: int = 8123
    database: str = "default"
    user: str = "default"
    password: str = ""

    @classmethod
    def from_env(cls) -> "DbConfig":
        return cls(
            host=os.environ.get("CLICKHOUSE_HOST", "127.0.0.1"),
            port=int(os.environ.get("CLICKHOUSE_PORT", "8123")),
            database=os.environ.get("CLICKHOUSE_DATABASE", "default"),
            user=os.environ.get("CLICKHOUSE_USER", "default"),
            password=os.environ.get("CLICKHOUSE_PASSWORD", ""),
        )


class Db:
    def __init__(self, cfg: DbConfig | None = None) -> None:
        self.cfg = cfg or DbConfig.from_env()
        self._client: Client | None = None

    @property
    def client(self) -> Client:
        if self._client is None:
            self._client = clickhouse_connect.get_client(
                host=self.cfg.host,
                port=self.cfg.port,
                database=self.cfg.database,
                username=self.cfg.user,
                password=self.cfg.password,
            )
        return self._client

    def query(self, sql: str, parameters: dict[str, tp.Any] | None = None) -> list[dict[str, tp.Any]]:
        result = self.client.query(sql, parameters=parameters or {})
        cols = result.column_names
        return [dict(zip(cols, row)) for row in result.result_rows]

    def query_one(self, sql: str, parameters: dict[str, tp.Any] | None = None) -> dict[str, tp.Any] | None:
        rows = self.query(sql, parameters)
        return rows[0] if rows else None

    def command(self, sql: str) -> tp.Any:
        return self.client.command(sql)
