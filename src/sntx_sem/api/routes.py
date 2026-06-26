"""FastAPI routes."""

from __future__ import annotations

from typing import Any, cast

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from sntx_sem import __version__
from sntx_sem.config import (
    bundled_database_status,
    resolve_embedding_provider,
)
from sntx_sem.search_service import HelpSearchService


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    domain: str = "all"
    limit: int = Field(default=5, ge=1, le=50)


def _get_service(request: Request) -> HelpSearchService:
    return cast(HelpSearchService, request.app.state.service)


def create_router() -> APIRouter:
    router = APIRouter()

    @router.get("/health")
    def health(request: Request) -> dict[str, Any]:
        cfg = request.app.state.config
        status = bundled_database_status(cfg)
        embedding = cfg.embedding
        return {
            "status": "ok",
            "version": __version__,
            "ready": status["ready"],
            "embedding": {
                "provider": resolve_embedding_provider(embedding),
                "model": embedding.model,
            },
        }

    @router.post("/search")
    def search(body: SearchRequest, request: Request) -> list[dict[str, Any]]:
        service = _get_service(request)
        return service.search(body.query, domain=body.domain, limit=body.limit)

    @router.get("/topic/{topic_id:path}")
    def get_topic(
        topic_id: str,
        request: Request,
        include_examples: bool = Query(default=True),
    ) -> dict[str, Any]:
        service = _get_service(request)
        topic = service.get_topic(topic_id, include_examples=include_examples)
        if topic is None:
            raise HTTPException(status_code=404, detail=f"Topic not found: {topic_id}")
        return topic

    @router.get("/stats")
    def stats(request: Request) -> dict[str, Any]:
        service = _get_service(request)
        return service.stats()

    return router


def create_ui_router(static_dir: Any) -> APIRouter:
    router = APIRouter()

    @router.get("/", include_in_schema=False)
    def index_page() -> FileResponse:
        return FileResponse(static_dir / "index.html")

    return router
