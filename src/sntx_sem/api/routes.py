"""FastAPI routes."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from sntx_sem import __version__
from sntx_sem.api.jobs import JobStore
from sntx_sem.api.logging_buffer import get_log_lines
from sntx_sem.config import (
    bundled_database_status,
    resolve_embedding_provider,
)
from sntx_sem.embeddings import create_embedding_backend
from sntx_sem.search_service import HelpSearchService


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    domain: str = "all"
    limit: int = Field(default=5, ge=1, le=50)


class ExamplesRequest(BaseModel):
    query: str = ""
    topic_id: str = ""
    limit: int = Field(default=5, ge=1, le=50)


class EmbeddingTestRequest(BaseModel):
    text: str = Field(default="тестовый запрос", min_length=1)


class IndexJobRequest(BaseModel):
    rebuild: bool = True


def _get_service(request: Request) -> HelpSearchService:
    return cast(HelpSearchService, request.app.state.service)


def _get_jobs(request: Request) -> JobStore:
    return cast(JobStore, request.app.state.jobs)


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

    @router.get("/status")
    def status(request: Request) -> dict[str, Any]:
        cfg = request.app.state.config
        return bundled_database_status(cfg)

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

    @router.post("/examples")
    def find_examples(body: ExamplesRequest, request: Request) -> list[dict[str, Any]]:
        if not body.query and not body.topic_id:
            raise HTTPException(status_code=400, detail="Provide query or topic_id")
        service = _get_service(request)
        return service.find_examples(
            query=body.query,
            topic_id=body.topic_id,
            limit=body.limit,
        )

    @router.get("/stats")
    def stats(request: Request) -> dict[str, Any]:
        service = _get_service(request)
        return service.stats()

    @router.get("/settings/embedding")
    def embedding_settings(request: Request) -> dict[str, Any]:
        cfg = request.app.state.config
        db_status = bundled_database_status(cfg)
        emb = cfg.embedding
        return {
            "provider": resolve_embedding_provider(emb),
            "model": emb.model,
            "device": emb.device,
            "base_url": emb.base_url or None,
            "api_key_set": bool(emb.resolved_api_key),
            "embedding_mismatch": db_status.get("index", {}).get("embedding_mismatch", False),
            "index_embedding_model": db_status.get("index", {}).get("embedding_model"),
        }

    @router.post("/settings/embedding/test")
    def embedding_test(body: EmbeddingTestRequest, request: Request) -> dict[str, Any]:
        cfg = request.app.state.config
        backend = create_embedding_backend(cfg.embedding)
        vector = backend.embed_query(body.text)
        return {
            "model": backend.model_id,
            "dimensions": len(vector),
        }

    @router.get("/logs")
    def logs(
        since: int = Query(default=0, ge=0), limit: int = Query(default=200, ge=1, le=500)
    ) -> dict[str, Any]:
        lines = get_log_lines(since=since, limit=limit)
        return {"lines": lines, "offset": since + len(lines)}

    @router.post("/jobs/ingest", status_code=202)
    def start_ingest_job(request: Request) -> dict[str, Any]:
        cfg = request.app.state.config
        job = _get_jobs(request).start_ingest(cfg)
        return job.to_detail()

    @router.post("/jobs/ingest-bsp", status_code=202)
    def start_ingest_bsp_job(request: Request) -> dict[str, Any]:
        cfg = request.app.state.config
        if not cfg.bsp.path:
            raise HTTPException(status_code=400, detail="bsp.path not configured")
        bsp_dir = Path(cfg.bsp.path)
        if not bsp_dir.is_dir():
            raise HTTPException(status_code=400, detail=f"BSP path not found: {bsp_dir}")
        job = _get_jobs(request).start_ingest_bsp(cfg, bsp_dir)
        return job.to_detail()

    @router.post("/jobs/index", status_code=202)
    def start_index_job(body: IndexJobRequest, request: Request) -> dict[str, Any]:
        cfg = request.app.state.config
        job = _get_jobs(request).start_index(cfg, rebuild=body.rebuild)
        return job.to_detail()

    @router.get("/jobs/{job_id}")
    def get_job(
        job_id: str, request: Request, since_log: int = Query(default=0, ge=0)
    ) -> dict[str, Any]:
        job = _get_jobs(request).get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
        return job.to_detail(since_log=since_log)

    return router


def create_ui_router(static_dir: Path) -> APIRouter:
    router = APIRouter()

    @router.get("/", include_in_schema=False)
    def index_page() -> FileResponse:
        return FileResponse(static_dir / "index.html")

    @router.get("/admin", include_in_schema=False)
    def admin_page() -> FileResponse:
        return FileResponse(static_dir / "admin.html")

    return router
