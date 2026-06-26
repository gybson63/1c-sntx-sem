"""Tests for MCP HTTP client."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from sntx_sem.mcp.client import SntxSemApiClient, SntxSemApiError


@pytest.fixture
def mock_client() -> MagicMock:
    with patch("sntx_sem.mcp.client.httpx.Client") as mock_cls:
        instance = MagicMock()
        mock_cls.return_value = instance
        yield instance


def test_client_requires_url(monkeypatch) -> None:
    monkeypatch.delenv("SNTX_SEM_API_URL", raising=False)
    with pytest.raises(ValueError, match="Base URL is required"):
        SntxSemApiClient(base_url="")


def test_health(mock_client: MagicMock) -> None:
    response = MagicMock()
    response.is_success = True
    response.content = b'{"status":"ok"}'
    response.json.return_value = {"status": "ok", "ready": True}
    mock_client.request.return_value = response

    client = SntxSemApiClient(base_url="http://test")
    assert client.health()["ready"] is True
    client.close()


def test_search(mock_client: MagicMock) -> None:
    response = MagicMock()
    response.is_success = True
    response.content = b"[{}]"
    response.json.return_value = [{"id": "t1", "title": "Test"}]
    mock_client.request.return_value = response

    client = SntxSemApiClient(base_url="http://test")
    items = client.search("query", domain="bsl", limit=3)
    assert items[0]["id"] == "t1"
    mock_client.request.assert_called_with(
        "POST",
        "/search",
        params=None,
        json={"query": "query", "domain": "bsl", "limit": 3},
    )
    client.close()


def test_api_error(mock_client: MagicMock) -> None:
    response = MagicMock()
    response.is_success = False
    response.status_code = 503
    response.text = "unavailable"
    response.json.side_effect = ValueError("not json")
    mock_client.request.return_value = response

    client = SntxSemApiClient(base_url="http://test")
    with pytest.raises(SntxSemApiError, match="503"):
        client.health()
    client.close()
