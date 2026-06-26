"""FastAPI application factory."""

from __future__ import annotations

import importlib.resources
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from sntx_sem import __version__
from sntx_sem.api.routes import create_router, create_ui_router
from sntx_sem.config import AppConfig
from sntx_sem.search_service import HelpSearchService


def _static_dir() -> Path:
    return Path(str(importlib.resources.files("sntx_sem.web") / "static"))


def create_app(config: AppConfig) -> FastAPI:
    service = HelpSearchService(config)
    app = FastAPI(
        title="1c-syntax-sem",
        description="Семантический поиск по справке платформы 1С",
        version=__version__,
    )
    app.state.config = config
    app.state.service = service

    static_dir = _static_dir()
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    app.include_router(create_router())
    app.include_router(create_ui_router(static_dir))

    return app
