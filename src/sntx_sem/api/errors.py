"""User-facing error messages for API and background jobs."""

from __future__ import annotations

import errno


def format_job_error(exc: BaseException) -> str:
    """Convert an exception into a short Russian message for Web-UI."""
    if isinstance(exc, OSError) and exc.errno == errno.ENOMEM:
        return (
            "Недостаточно памяти при построении индекса. "
            "Увеличьте RAM для Docker Desktop или выберите модель e5-small, "
            "затем выполните Rebuild Index."
        )
    if isinstance(exc, MemoryError):
        return (
            "Недостаточно памяти при построении индекса. "
            "Увеличьте RAM или используйте более лёгкую модель эмбеддингов."
        )
    if isinstance(exc, FileNotFoundError):
        return f"Файл не найден: {exc.filename or exc}"
    text = str(exc).strip()
    if not text:
        return exc.__class__.__name__
    if len(text) > 500:
        return text[:500] + "…"
    return text
