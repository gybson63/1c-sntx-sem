"""HTTP client for sntx-sem API (used by thin MCP server)."""

from __future__ import annotations

import json
import os
from typing import Any, cast
from urllib.parse import quote

import httpx

DEFAULT_TIMEOUT = 60.0
ENV_API_URL = "SNTX_SEM_API_URL"
ENV_API_TIMEOUT = "SNTX_SEM_API_TIMEOUT"
DEFAULT_API_URL = "http://127.0.0.1:8051"


class SntxSemApiError(Exception):
    """API request failed."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class SntxSemApiClient:
    """Thin wrapper over sntx-sem REST endpoints."""

    def __init__(
        self,
        base_url: str | None = None,
        *,
        timeout: float | None = None,
    ) -> None:
        raw_url = base_url if base_url is not None else os.environ.get(ENV_API_URL, DEFAULT_API_URL)
        url = raw_url.strip().rstrip("/")
        if not url:
            msg = f"Base URL is required. Set {ENV_API_URL} (e.g. http://localhost:8051)."
            raise ValueError(msg)

        timeout_value = timeout
        if timeout_value is None:
            raw_timeout = os.environ.get(ENV_API_TIMEOUT, "")
            timeout_value = float(raw_timeout) if raw_timeout else DEFAULT_TIMEOUT

        self.base_url = url
        self._client = httpx.Client(base_url=url, timeout=timeout_value)

    def close(self) -> None:
        self._client.close()

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> Any:
        response = self._client.request(method, path, params=params, json=json_body)
        if response.is_success:
            if not response.content:
                return {}
            return response.json()

        detail = response.text
        try:
            payload = response.json()
            if isinstance(payload, dict) and "detail" in payload:
                detail = str(payload["detail"])
        except (json.JSONDecodeError, ValueError):
            pass
        raise SntxSemApiError(
            f"{method} {path} failed ({response.status_code}): {detail}",
            status_code=response.status_code,
        )

    def health(self) -> dict[str, Any]:
        return cast(dict[str, Any], self._request("GET", "/health"))

    def status(self) -> dict[str, Any]:
        return cast(dict[str, Any], self._request("GET", "/status"))

    def search(
        self,
        query: str,
        *,
        domain: str = "all",
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        body = {"query": query, "domain": domain, "limit": limit}
        result = self._request("POST", "/search", json_body=body)
        return cast(list[dict[str, Any]], result)

    def get_topic(
        self,
        topic_id: str,
        *,
        include_examples: bool = True,
    ) -> dict[str, Any]:
        params = {"include_examples": include_examples}
        path = f"/topic/{quote(topic_id, safe='')}"
        return cast(dict[str, Any], self._request("GET", path, params=params))

    def find_examples(
        self,
        *,
        query: str = "",
        topic_id: str = "",
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        body: dict[str, Any] = {"limit": limit}
        if query:
            body["query"] = query
        if topic_id:
            body["topic_id"] = topic_id
        result = self._request("POST", "/examples", json_body=body)
        return cast(list[dict[str, Any]], result)

    def stats(self) -> dict[str, Any]:
        return cast(dict[str, Any], self._request("GET", "/stats"))
