"""OpenAI-compatible LLM client."""

from __future__ import annotations

import json
import os
import re
from typing import Any

import httpx


class LLMClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.timeout = timeout

    def chat(self, messages: list[dict[str, str]], temperature: float = 0.1) -> str:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]

    def chat_json(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        text = self.chat(messages)
        return parse_json_response(text)


def parse_json_response(text: str) -> dict[str, Any]:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    return json.loads(text)


def resolve_model_for_task(task: str, config_path: str | None = None) -> tuple[str, str]:
    """Return (model_id, base_url) for a benchmark task."""
    import yaml
    from pathlib import Path

    path = Path(config_path or "config/benchmark_results.yaml")
    if not path.is_file():
        return "deepseek-chat", "https://api.deepseek.com/v1"

    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    model_key = data.get("tasks", {}).get(task, "deepseek-chat")
    model_cfg = data.get("models", {}).get(model_key, {})
    return model_cfg.get("model_id", model_key), model_cfg.get(
        "base_url", "https://api.deepseek.com/v1"
    )
