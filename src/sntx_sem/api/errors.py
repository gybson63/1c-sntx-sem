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
        "Увеличьте RAM для Docker Desktop (рекомендуется 8 ГБ+) или задайте "
        "search.build_vector_index: false в config.yaml, затем выполните Rebuild Index."
    )


def _ivf_oom_message() -> str:
    return (
        "Эмбеддинги завершены, но нехватка памяти на построении IVF-индекса LanceDB. "
        "Увеличьте RAM Docker или задайте search.build_vector_index: false в config.yaml."
    )


def _search_oom_message() -> str:
    return (
        "Недостаточно памяти при поиске. Увеличьте RAM Docker (рекомендуется 8 ГБ+), "
        "выберите конкретный домен вместо «Все домены» или уменьшите search.dense_top_k / "
        "search.bm25_top_k в config.yaml."
    )


def format_search_error(exc: BaseException) -> str:
    """Convert a search exception into a short Russian message for Web-UI."""
    if isinstance(exc, ZeroDivisionError):
        return (
            "Домен не проиндексирован (нет чанков в индексе). "
            "Выполните Ingest BSP + Rebuild Index в /admin."
        )
    if isinstance(exc, OSError) and exc.errno == errno.ENOMEM:
        return _search_oom_message()
    if isinstance(exc, MemoryError):
        return _search_oom_message()

    text = str(exc).strip()
    lowered = text.lower()
    if any(marker in lowered for marker in _OOM_MARKERS):
        return _search_oom_message()
    if "RepositoryNotFoundError" in text or "text-embedding-3-small" in text:
        return (
            "Не удалось загрузить модель эмбеддингов. В /admin укажите "
            "intfloat/multilingual-e5-base (локальный E5) или выполните Rebuild Index "
            "под текущей моделью в config.yaml."
        )
    if "ReadTimeout" in text or "ConnectTimeout" in text:
        return (
            "Таймаут API эмбеддингов при поиске. Увеличьте embedding.timeout в config "
            "или переключитесь на локальную модель E5."
        )
    if "Chunks not found" in text or "Run ingest first" in text:
        return "База не собрана: выполните Ingest HBK + Index в /admin."
    if "lance" in lowered and ("not found" in lowered or "no such file" in lowered):
        return (
            "Индекс LanceDB повреждён или неполный после rebuild. "
            "Откройте /admin → Rebuild Index. В Docker с ~4 ГБ RAM задайте "
            "search.build_vector_index: false в config.yaml."
        )
    if not text:
        return exc.__class__.__name__
    if len(text) > 400:
        return text[:400] + "…"
    return text


def format_job_error(exc: BaseException, *, traceback_text: str = "") -> str:
    """Convert an exception into a short Russian message for Web-UI."""
    if isinstance(exc, OSError) and exc.errno == errno.ENOMEM:
        return _memory_error_message()
    if isinstance(exc, MemoryError):
        return _memory_error_message()

    combined = f"{traceback_text}\n{exc}".lower()
    if "create_index" in combined and any(marker in combined for marker in _OOM_MARKERS):
        return _ivf_oom_message()

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
