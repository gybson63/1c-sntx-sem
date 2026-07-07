"""Tests for HTTP API."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml
from fastapi.testclient import TestClient

from sntx_sem.api.app import create_app
from sntx_sem.config import AppConfig, load_config


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
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
            "excerpt_start": 0,
            "excerpt_end": 9,
            "highlight_terms": ["соединение"],
            "match_sources": ["semantic", "bm25"],
            "match_explanation": "Семантический поиск: rank 1; BM25: rank 2.",
            "semantic_excerpt": "Описание семантического совпадения.",
            "semantic_highlight_terms": ["семантического"],
            "score_breakdown": {
                "total": 0.9,
                "dense_rank": 1,
                "bm25_rank": 2,
                "dense_similarity": 0.8,
            },
        }
    ]
    mock_service.get_topic.return_value = {
        "id": "test:1",
        "title": "Левое соединение",
        "domain": "query_lang",
        "text": "Полный текст",
    }
    mock_service.stats.return_value = {"query_lang": 10, "examples": 0}
    mock_service.find_examples.return_value = [{"id": "ex1", "title": "Example"}]
    app.state.service = mock_service

    real_status = None

    def ready_status(config: AppConfig) -> dict:
        from sntx_sem.config import bundled_database_status

        nonlocal real_status
        if real_status is None:
            real_status = bundled_database_status(config)
        status = dict(real_status)
        status["ready"] = True
        return status

    monkeypatch.setattr("sntx_sem.api.routes.bundled_database_status", ready_status)
    return TestClient(app)


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in {"ok", "degraded"}
    assert data["ready"] is True
    assert "version" in data
    assert "issues" in data
    assert isinstance(data["issues"], list)
    assert data["embedding"]["model"] == "intfloat/multilingual-e5-base"
    assert "index" in data


def test_search(client: TestClient) -> None:
    response = client.post("/search", json={"query": "соединение", "domain": "query", "limit": 5})
    assert response.status_code == 200
    items = response.json()
    assert len(items) == 1
    assert items[0]["title"] == "Левое соединение"
    assert items[0]["highlight_terms"] == ["соединение"]
    assert items[0]["match_sources"] == ["semantic", "bm25"]
    assert items[0]["score_breakdown"]["dense_rank"] == 1
    assert items[0]["semantic_excerpt"] == "Описание семантического совпадения."


def test_search_bsp_empty_domain_returns_empty(client: TestClient) -> None:
    client.app.state.service.search.return_value = []
    response = client.post(
        "/search", json={"query": "строку в массив", "domain": "bsp", "limit": 5}
    )
    assert response.status_code == 200
    assert response.json() == []


def test_search_not_ready(tmp_path: Path) -> None:
    cfg = AppConfig(data_dir=tmp_path / "data", index_dir=tmp_path / "index")
    app = create_app(cfg)
    client = TestClient(app)
    response = client.post("/search", json={"query": "тест", "domain": "all", "limit": 5})
    assert response.status_code == 503
    assert "не готов" in response.json()["detail"].lower()


def test_search_service_error(client: TestClient) -> None:
    client.app.state.service.search.side_effect = RuntimeError("text-embedding-3-small boom")
    response = client.post("/search", json={"query": "тест", "domain": "all", "limit": 5})
    assert response.status_code == 503
    assert "эмбеддинг" in response.json()["detail"].lower()


def test_search_lance_oom_error(client: TestClient) -> None:
    client.app.state.service.search.side_effect = RuntimeError(
        "lance error: LanceError(IO): Cannot allocate memory (os error 12), "
        "library/core/src/ops/function.rs:250:5"
    )
    response = client.post("/search", json={"query": "СтрСоединить", "domain": "all", "limit": 5})
    assert response.status_code == 503
    detail = response.json()["detail"].lower()
    assert "памят" in detail
    assert "lance error" not in detail


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


def test_status(client: TestClient) -> None:
    response = client.get("/status")
    assert response.status_code == 200
    data = response.json()
    assert "ready" in data
    assert "config" in data
    assert "issues" in data
    assert isinstance(data["issues"], list)


def test_examples(client: TestClient) -> None:
    response = client.post("/examples", json={"topic_id": "test:1", "limit": 3})
    assert response.status_code == 200
    assert response.json()[0]["id"] == "ex1"


def test_examples_validation(client: TestClient) -> None:
    response = client.post("/examples", json={"limit": 3})
    assert response.status_code == 400


def test_embedding_settings(client: TestClient) -> None:
    response = client.get("/settings/embedding")
    assert response.status_code == 200
    data = response.json()
    assert data["model"] == "intfloat/multilingual-e5-base"
    assert "providers" in data
    assert data["config_writable"] is False


def test_update_embedding_settings(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        yaml.safe_dump(
            {
                "embedding": {
                    "provider": "sentence_transformers",
                    "model": "intfloat/multilingual-e5-small",
                }
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    cfg = load_config(config_file)
    app = create_app(cfg)
    app.state.service = MagicMock()
    client = TestClient(app)

    response = client.put(
        "/settings/embedding",
        json={
            "provider": "openai_compatible",
            "model": "text-embedding-3-small",
            "base_url": "https://example.com/v1",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["provider"] == "openai_compatible"
    assert data["model"] == "text-embedding-3-small"
    assert data["saved"] is True

    saved = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    assert saved["embedding"]["provider"] == "openai_compatible"
    assert saved["embedding"]["model"] == "text-embedding-3-small"


def test_update_embedding_settings_coerces_local_model(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        yaml.safe_dump(
            {
                "embedding": {
                    "provider": "openai_compatible",
                    "model": "text-embedding-3-small",
                    "base_url": "https://example.com/v1",
                }
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    cfg = load_config(config_file)
    app = create_app(cfg)
    app.state.service = MagicMock()
    client = TestClient(app)

    response = client.put(
        "/settings/embedding",
        json={
            "provider": "sentence_transformers",
            "model": "text-embedding-3-small",
            "device": "cpu",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["provider"] == "sentence_transformers"
    assert data["model"] == "intfloat/multilingual-e5-base"
    assert data["model_adjustment"]

    saved = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    assert saved["embedding"]["model"] == "intfloat/multilingual-e5-base"


def test_update_embedding_settings_invalid_provider(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text("embedding:\n  model: test\n", encoding="utf-8")
    cfg = load_config(config_file)
    app = create_app(cfg)
    client = TestClient(app)

    response = client.put("/settings/embedding", json={"provider": "unknown"})
    assert response.status_code == 400


def test_logs(client: TestClient) -> None:
    response = client.get("/logs")
    assert response.status_code == 200
    assert "lines" in response.json()


def test_admin_page(client: TestClient) -> None:
    response = client.get("/admin")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
