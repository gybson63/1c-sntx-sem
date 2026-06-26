"""In-memory ring buffer for API log tail."""

from __future__ import annotations

import logging
import threading
from collections import deque

_MAX_LINES = 500
_buffer: deque[str] = deque(maxlen=_MAX_LINES)
_lock = threading.Lock()


class RingBufferHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
        except Exception:
            self.handleError(record)
            return
        with _lock:
            _buffer.append(msg)


def install_api_logging() -> None:
    root = logging.getLogger("sntx_sem")
    handler = RingBufferHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root.addHandler(handler)
    if root.level == logging.NOTSET:
        root.setLevel(logging.INFO)


def get_log_lines(*, since: int = 0, limit: int = 200) -> list[str]:
    with _lock:
        lines = list(_buffer)
    if since > 0:
        lines = lines[since:]
    return lines[-limit:]
