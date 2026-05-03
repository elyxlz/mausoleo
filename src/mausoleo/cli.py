"""Mausoleo CLI.

Wraps the search API server. Every command outputs JSON to stdout and nothing
else (no colour, no spinners, no human-friendly tables) — the consumer is an
LLM agent that needs predictable structured data.

Server URL precedence: ``--server`` flag > ``MAUSOLEO_SERVER_URL`` env var >
``http://127.0.0.1:8000``.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sys
import typing as tp

import httpx
import typer

DEFAULT_SERVER = "http://127.0.0.1:8000"

app = typer.Typer(name="mausoleo", no_args_is_help=True, add_completion=False)
search_app = typer.Typer(name="search", no_args_is_help=True, add_completion=False)
app.add_typer(search_app, name="search", help="Search the index (semantic / text / hybrid).")


def _server_url(override: str | None) -> str:
    return override or os.environ.get("MAUSOLEO_SERVER_URL", DEFAULT_SERVER)


def _print(payload: tp.Any) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False, default=_json_default))
    sys.stdout.write("\n")
    sys.stdout.flush()


def _json_default(obj: tp.Any) -> tp.Any:
    if isinstance(obj, (dt.date, dt.datetime)):
        return obj.isoformat()
    raise TypeError(f"unserializable: {type(obj)!r}")


def _get(server: str, path: str, params: dict[str, tp.Any] | None = None) -> tp.Any:
    with httpx.Client(timeout=60) as cli:
        r = cli.get(server.rstrip("/") + path, params=params or {})
    if r.status_code >= 400:
        _print({"error": r.text, "status": r.status_code})
        raise typer.Exit(code=2)
    return r.json()


def _post(server: str, path: str, body: dict[str, tp.Any]) -> tp.Any:
    with httpx.Client(timeout=120) as cli:
        r = cli.post(server.rstrip("/") + path, json=body)
    if r.status_code >= 400:
        _print({"error": r.text, "status": r.status_code})
        raise typer.Exit(code=2)
    return r.json()


# ---------------------------------------------------------------------------
# Tree navigation commands
# ---------------------------------------------------------------------------

@app.command()
def root(
    server: str | None = typer.Option(None, "--server", help="API server URL"),
) -> None:
    """Get the archive root node — entry point for top-down traversal."""
    _print(_get(_server_url(server), "/root"))


@app.command()
def node(
    node_id: str,
    server: str | None = typer.Option(None, "--server"),
) -> None:
    """Get a single node's details by id."""
    _print(_get(_server_url(server), f"/nodes/{node_id}"))


@app.command()
def children(
    node_id: str,
    offset: int = typer.Option(0, "--offset"),
    limit: int = typer.Option(100, "--limit"),
    server: str | None = typer.Option(None, "--server"),
) -> None:
    """List a node's direct children, ordered by position."""
    _print(
        _get(
            _server_url(server),
            f"/nodes/{node_id}/children",
            params={"offset": offset, "limit": limit},
        )
    )


@app.command()
def parent(
    node_id: str,
    server: str | None = typer.Option(None, "--server"),
) -> None:
    """Walk up the tree."""
    _print(_get(_server_url(server), f"/nodes/{node_id}/parent"))


@app.command()
def text(
    node_id: str,
    server: str | None = typer.Option(None, "--server"),
) -> None:
    """Get raw text — leaf nodes return their paragraph; non-leaves are reconstructed."""
    _print(_get(_server_url(server), f"/nodes/{node_id}/text"))


@app.command()
def stats(
    server: str | None = typer.Option(None, "--server"),
) -> None:
    """Index statistics — total nodes per level, date range, sources."""
    _print(_get(_server_url(server), "/stats"))


# ---------------------------------------------------------------------------
# Search commands
# ---------------------------------------------------------------------------

def _search_body(
    query: str,
    level: str | None,
    date_from: str | None,
    date_to: str | None,
    limit: int,
) -> dict[str, tp.Any]:
    return {
        "query": query,
        "level": level,
        "date_start": date_from,
        "date_end": date_to,
        "limit": limit,
    }


@search_app.command("semantic")
def search_semantic(
    query: str,
    level: str | None = typer.Option(None, "--level"),
    date_from: str | None = typer.Option(None, "--from"),
    date_to: str | None = typer.Option(None, "--to"),
    limit: int = typer.Option(20, "--limit"),
    server: str | None = typer.Option(None, "--server"),
) -> None:
    """Vector-similarity search across the index."""
    _print(
        _post(
            _server_url(server),
            "/search/semantic",
            _search_body(query, level, date_from, date_to, limit),
        )
    )


@search_app.command("text")
def search_text(
    query: str,
    level: str | None = typer.Option(None, "--level"),
    date_from: str | None = typer.Option(None, "--from"),
    date_to: str | None = typer.Option(None, "--to"),
    limit: int = typer.Option(20, "--limit"),
    server: str | None = typer.Option(None, "--server"),
) -> None:
    """Full-text / keyword search."""
    _print(
        _post(
            _server_url(server),
            "/search/text",
            _search_body(query, level, date_from, date_to, limit),
        )
    )


@search_app.command("hybrid")
def search_hybrid(
    query: str,
    level: str | None = typer.Option(None, "--level"),
    date_from: str | None = typer.Option(None, "--from"),
    date_to: str | None = typer.Option(None, "--to"),
    limit: int = typer.Option(20, "--limit"),
    server: str | None = typer.Option(None, "--server"),
) -> None:
    """RRF combination of semantic + text rankings."""
    _print(
        _post(
            _server_url(server),
            "/search/hybrid",
            _search_body(query, level, date_from, date_to, limit),
        )
    )


# ---------------------------------------------------------------------------
# Loader sub-app — exposed so users can populate ClickHouse from the CLI.
# ---------------------------------------------------------------------------

@app.command("load")
def load_cmd(
    transcription_dir: str = typer.Option(
        "/tmp/mausoleo/eval/transcriptions",
        "--transcriptions",
        help="Directory of OCR transcription JSONs",
    ),
    summary_dir: str = typer.Option(
        "/tmp/mausoleo/eval/summaries",
        "--summaries",
        help="Directory of per-level summary JSONs",
    ),
    date_from: str | None = typer.Option(None, "--from"),
    date_to: str | None = typer.Option(None, "--to"),
    truncate: bool = typer.Option(True, "--truncate/--no-truncate"),
    host: str = typer.Option(
        os.environ.get("CLICKHOUSE_HOST", "127.0.0.1"), "--host"
    ),
    port: int = typer.Option(int(os.environ.get("CLICKHOUSE_PORT", "8123")), "--port"),
    database: str = typer.Option(
        os.environ.get("CLICKHOUSE_DATABASE", "default"), "--database"
    ),
) -> None:
    """Load transcriptions + summaries into ClickHouse."""
    import pathlib as pl

    from mausoleo.index import loader

    cfg = loader.LoaderConfig(
        transcription_dir=pl.Path(transcription_dir),
        summary_dir=pl.Path(summary_dir),
        date_start=dt.date.fromisoformat(date_from) if date_from else None,
        date_end=dt.date.fromisoformat(date_to) if date_to else None,
    )
    result = loader.load(
        cfg,
        host=host,
        port=port,
        database=database,
        truncate=truncate,
    )
    _print(result)


if __name__ == "__main__":
    app()
