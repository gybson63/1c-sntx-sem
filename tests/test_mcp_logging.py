"""Tests for MCP logging helpers."""

from sntx_sem.mcp_logging import truncate_for_log


def test_truncate_for_log_short_value() -> None:
    assert truncate_for_log({"q": "test"}, max_chars=100) == '{"q": "test"}'


def test_truncate_for_log_long_value() -> None:
    text = truncate_for_log("x" * 50, max_chars=10)
    assert "truncated" in text
    assert "52 chars" in text
    assert len(text) < 52
