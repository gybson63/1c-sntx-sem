"""Tests for HTTP API."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from sntx_sem.api.app import create_app
from sntx_sem.config import AppConfig


@pytest.fixture
def client() -> TestClient:
    cfg = AppConfig()
    app = create_app(cfg)
    mock_service = MagicMock()
    mock_service.search.return_value = [
        {
            "id": "test:1",
            "domain": "query_lang",
            "title": "Левое соединение",
            "score": 0.9,
            "entity_kind": "topic",
            "html_path": "",
            "excerpt": "Описание…",
        }
    ]
    mock_service.get_topic.return_value = {
        "id": "test:1",
        "title": "Левое соединение",
        "domain": "query_lang",
        "text": "Полный текст",
    }
    mock_service.stats.return_value = {"query_lang": 10, "examples": 0}
    app.state.service = mock_service
    return TestClient(app)


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data
    assert data["embedding"]["model"] == "intfloat/multilingual-e5-small"


def test_search(client: TestClient) -> None:
    response = client.post("/search", json={"query": "соединение", "domain": "query", "limit": 5})
    assert response.status_code == 200
    items = response.json()
    assert len(items) == 1
    assert items[0]["title"] == "Левое соединение"


def test_get_topic(client: TestClient) -> None:
    response = client.get("/topic/test:1")
    assert response.status_code == 200
    assert response.json()["text"] == "Полный текст"


def test_get_topic_not_found(client: TestClient) -> None:
    client.app.state.service.get_topic.return_value = None
    response = client.get("/topic/missing")
    assert response.status_code == 404


def test_stats(client: TestClient) -> None:
    response = client.get("/stats")
    assert response.status_code == 200
    assert response.json()["query_lang"] == 10


def test_index_page(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
