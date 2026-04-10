from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from mausoleo.ocr.config import OcrPipelineConfig
from mausoleo.ocr.operators import LlmCleanup, ParseIssue, VlmOcr
from mausoleo.server.app import create_app


FAKE_JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 100

MOCK_CONFIG = OcrPipelineConfig(
    name="mock_test",
    operators=[VlmOcr(mock=True), LlmCleanup(mock=True), ParseIssue()],
)


@pytest.fixture
def client() -> TestClient:
    app = create_app(config=MOCK_CONFIG)
    return TestClient(app)


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ocr_endpoint(client: TestClient) -> None:
    files = [("files", (f"{i}.jpeg", FAKE_JPEG, "image/jpeg")) for i in range(4)]
    response = client.post("/ocr", files=files, data={"date": "1885-06-15"})

    assert response.status_code == 200
    data = response.json()
    assert data["date"] == "1885-06-15"
    assert data["page_count"] == 4
    assert len(data["articles"]) == 4


def test_ocr_no_config() -> None:
    app = create_app(config=None)
    client = TestClient(app)
    files = [("files", ("1.jpeg", FAKE_JPEG, "image/jpeg"))]
    response = client.post("/ocr", files=files, data={"date": "1885-06-15"})
    assert response.status_code == 500
