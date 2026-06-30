"""User-facing error messages for API and background jobs."""

from __future__ import annotations

import errno

_OOM_MARKERS = (
    "cannot allocate memory",
    "os error 12",
    "out of memory",
    "ENOMEM",
)


def _memory_error_message() -> str:
    return (
        "Недостаточно памяти при построении индекса. "
        "Увеличьте RAM для Docker Desktop (рекомендуется 8 ГБ+) или выберите модель e5-small, "
        "затем выполните Rebuild Index."
    )


def format_job_error(exc: BaseException) -> str:
    """Convert an exception into a short Russian message for Web-UI."""
    if isinstance(exc, OSError) and exc.errno == errno.ENOMEM:
        return _memory_error_message()
    if isinstance(exc, MemoryError):
        return _memory_error_message()
    if isinstance(exc, FileNotFoundError):
        return f"Файл не найден: {exc.filename or exc}"
    text = str(exc).strip()
    if any(marker in text.lower() for marker in _OOM_MARKERS):
        return _memory_error_message()
    if not text:
        return exc.__class__.__name__
    if len(text) > 500:
        return text[:500] + "…"
    return text
