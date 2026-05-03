"""End-to-end smoke tests for the Mausoleo search stack.

These tests assume:

- A ClickHouse server is reachable (host/port from ``CLICKHOUSE_HOST`` /
  ``CLICKHOUSE_PORT``, defaulting to localhost:8123).
- The transcription corpus is at ``/tmp/mausoleo/eval/transcriptions``.

The tests boot the FastAPI app with FastAPI's ``TestClient`` (no separate
uvicorn process), so they don't need the server running on a port.
"""
from __future__ import annotations

import datetime as dt
import os
import pathlib as pl

import pytest

clickhouse_connect = pytest.importorskip("clickhouse_connect")
fastapi_testclient = pytest.importorskip("fastapi.testclient")


def _ch_available() -> bool:
    try:
        c = clickhouse_connect.get_client(
            host=os.environ.get("CLICKHOUSE_HOST", "127.0.0.1"),
            port=int(os.environ.get("CLICKHOUSE_PORT", "8123")),
        )
        c.command("SELECT 1")
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _ch_available(),
    reason="ClickHouse not reachable; skipping integration tests",
)


@pytest.fixture(scope="session")
def loaded_db() -> str:
    """Load a small slice of the corpus and return the database name."""
    from mausoleo.index import loader

    transcriptions = pl.Path(
        os.environ.get("MAUSOLEO_TX_DIR", "/tmp/mausoleo/eval/transcriptions")
    )
    if not transcriptions.exists():
        pytest.skip(f"no transcriptions at {transcriptions}")
    cfg = loader.LoaderConfig(
        transcription_dir=transcriptions,
        date_start=dt.date(1943, 7, 1),
        date_end=dt.date(1943, 7, 5),
    )
    loader.load(cfg, truncate=True)
    return "default"


@pytest.fixture
def client(loaded_db: str) -> "fastapi_testclient.TestClient":
    os.environ["MAUSOLEO_EMBED_BACKEND"] = "zero"
    from mausoleo.server.app import create_app

    app = create_app()
    return fastapi_testclient.TestClient(app)


def test_health(client: "fastapi_testclient.TestClient") -> None:
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_root(client: "fastapi_testclient.TestClient") -> None:
    r = client.get("/root")
    assert r.status_code == 200
    body = r.json()
    assert body["node_id"] == "archive"
    assert body["level"] == "archive"


def test_stats(client: "fastapi_testclient.TestClient") -> None:
    r = client.get("/stats")
    assert r.status_code == 200
    body = r.json()
    assert body["total"] > 0
    levels = {row["level"]: row["count"] for row in body["by_level"]}
    assert levels.get("day", 0) >= 5
    assert levels.get("paragraph", 0) > 0


def test_drill_down(client: "fastapi_testclient.TestClient") -> None:
    """root → month → day → article → paragraph chain."""
    r = client.get("/root")
    assert r.status_code == 200

    r = client.get("/nodes/1943-07")
    assert r.status_code == 200
    assert r.json()["level"] == "month"

    r = client.get("/nodes/1943-07/children", params={"limit": 100})
    assert r.status_code == 200
    days = r.json()["children"]
    assert len(days) >= 5
    a_day = days[0]["node_id"]

    r = client.get(f"/nodes/{a_day}/children", params={"limit": 5})
    assert r.status_code == 200
    articles = r.json()["children"]
    assert len(articles) > 0
    art_id = articles[0]["node_id"]

    r = client.get(f"/nodes/{art_id}/text")
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body.get("text"), str)
    assert body["paragraph_count"] >= 1


def test_parent(client: "fastapi_testclient.TestClient") -> None:
    r = client.get("/nodes/1943-07-01/parent")
    assert r.status_code == 200
    assert r.json()["node_id"] == "1943-07"


def test_text_search(client: "fastapi_testclient.TestClient") -> None:
    r = client.post(
        "/search/text",
        json={"query": "Mussolini", "limit": 5},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["mode"] == "text"
    assert len(body["results"]) > 0


def test_semantic_search(client: "fastapi_testclient.TestClient") -> None:
    r = client.post(
        "/search/semantic",
        json={"query": "guerra", "limit": 5},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["mode"] == "semantic"
    # With zero embeddings everything ties on distance 0, but we should still
    # get results back.
    assert len(body["results"]) > 0


def test_hybrid_search(client: "fastapi_testclient.TestClient") -> None:
    r = client.post(
        "/search/hybrid",
        json={"query": "Roma", "limit": 5},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["mode"] == "hybrid"
    assert len(body["results"]) > 0


def test_recursive_text_for_day(client: "fastapi_testclient.TestClient") -> None:
    """Day-level text should reconstruct from descendant paragraphs."""
    r = client.get("/nodes/1943-07-01/text")
    assert r.status_code == 200
    body = r.json()
    assert body["paragraph_count"] > 10
    assert len(body["text"]) > 1000


def test_loader_idempotent() -> None:
    """Building the node list is deterministic — same ids, same counts."""
    from mausoleo.index import loader

    cfg = loader.LoaderConfig(
        transcription_dir=pl.Path("/tmp/mausoleo/eval/transcriptions"),
        date_start=dt.date(1943, 7, 1),
        date_end=dt.date(1943, 7, 2),
    )
    a = loader.build_nodes(cfg)
    b = loader.build_nodes(cfg)
    assert len(a) == len(b)
    assert sorted(n.node_id for n in a) == sorted(n.node_id for n in b)
