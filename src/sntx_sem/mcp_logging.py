"""MCP tool call logging (stderr / optional file)."""

from __future__ import annotations

import json
import logging
import os
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

_LOGGER_NAME = "sntx_sem.mcp"
_DEFAULT_MAX_CHARS = 2000


def _env_log_level() -> int:
    level_name = os.environ.get("SNTX_SEM_MCP_LOG_LEVEL", "INFO").upper()
    return getattr(logging, level_name, logging.INFO)


def _env_max_chars() -> int:
    raw = os.environ.get("SNTX_SEM_MCP_LOG_MAX_CHARS", str(_DEFAULT_MAX_CHARS))
    try:
        return max(0, int(raw))
    except ValueError:
        return _DEFAULT_MAX_CHARS


def _configure_logger() -> logging.Logger:
    logger = logging.getLogger(_LOGGER_NAME)
    if logger.handlers:
        return logger

    logger.setLevel(_env_log_level())
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stderr_handler = logging.StreamHandler()
    stderr_handler.setFormatter(formatter)
    logger.addHandler(stderr_handler)

    log_file = os.environ.get("SNTX_SEM_MCP_LOG_FILE", "").strip()
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger


def truncate_for_log(value: Any, *, max_chars: int | None = None) -> str:
    """Serialize value for logs, truncating long payloads."""
    limit = _DEFAULT_MAX_CHARS if max_chars is None else max_chars
    try:
        text = json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        text = repr(value)

    if limit == 0:
        return text
    if len(text) <= limit:
        return text
    return f"{text[:limit]}… [truncated, total {len(text)} chars]"


def install_mcp_logging(mcp: FastMCP[Any]) -> None:
    """Log MCP tool requests and responses via ToolManager hook."""
    logger = _configure_logger()
    manager = mcp._tool_manager
    original_call = manager.call_tool
    max_chars = _env_max_chars()

    async def logged_call_tool(
        name: str,
        arguments: dict[str, Any],
        context: Any = None,
        convert_result: bool = False,
    ) -> Any:
        logger.info(
            "request tool=%s arguments=%s",
            name,
            truncate_for_log(arguments, max_chars=max_chars),
        )
        started = time.perf_counter()
        try:
            result = await original_call(
                name,
                arguments,
                context=context,
                convert_result=convert_result,
            )
        except Exception:
            elapsed_ms = (time.perf_counter() - started) * 1000
            logger.exception("error tool=%s duration_ms=%.1f", name, elapsed_ms)
            raise

        elapsed_ms = (time.perf_counter() - started) * 1000
        logger.info(
            "response tool=%s duration_ms=%.1f result=%s",
            name,
            elapsed_ms,
            truncate_for_log(result, max_chars=max_chars),
        )
        return result

    manager.call_tool = logged_call_tool  # type: ignore[method-assign]
