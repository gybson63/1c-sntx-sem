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
    AppConfig,
    bundled_database_status,
    embedding_settings_view,
    resolve_embedding_provider,
    save_embedding_settings,
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


class EmbeddingSettingsUpdate(BaseModel):
    provider: str | None = None
    model: str | None = None
    device: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    api_key_env: str | None = None
    query_prefix: str | None = None
    passage_prefix: str | None = None


class IndexJobRequest(BaseModel):
    rebuild: bool = True


def _format_search_error(exc: Exception) -> str:
    text = str(exc)
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
    if len(text) > 400:
        return text[:400] + "…"
    return text or exc.__class__.__name__


def _get_service(request: Request) -> HelpSearchService:
    return cast(HelpSearchService, request.app.state.service)


def _get_jobs(request: Request) -> JobStore:
    return cast(JobStore, request.app.state.jobs)


def _apply_config(request: Request, cfg: AppConfig) -> None:
    request.app.state.config = cfg
    _get_service(request).update_config(cfg)


def create_router() -> APIRouter:
    router = APIRouter()

    @router.get("/health")
    def health(request: Request) -> dict[str, Any]:
        cfg = request.app.state.config
        status = bundled_database_status(cfg)
        embedding = cfg.embedding
        index_info = status.get("index", {})
        return {
            "status": "ok" if status["ready"] else "degraded",
            "version": __version__,
            "ready": status["ready"],
            "issues": status.get("issues", []),
            "embedding_in_sync": status.get("embedding_in_sync", True),
            "embedding": {
                "provider": resolve_embedding_provider(embedding),
                "model": embedding.model,
            },
            "index": {
                "embedding_model": index_info.get("embedding_model"),
                "embedding_provider": index_info.get("embedding_provider"),
                "indexed_chunks": index_info.get("indexed_chunks", 0),
            },
        }

    @router.get("/status")
    def status(request: Request) -> dict[str, Any]:
        cfg = request.app.state.config
        return bundled_database_status(cfg)

    @router.post("/search")
    def search(body: SearchRequest, request: Request) -> list[dict[str, Any]]:
        db_status = bundled_database_status(request.app.state.config)
        if not db_status.get("ready"):
            raise HTTPException(
                status_code=503,
                detail="Индекс не готов. Откройте /admin и выполните Ingest HBK + Index.",
            )
        service = _get_service(request)
        try:
            return service.search(body.query, domain=body.domain, limit=body.limit)
        except Exception as exc:
            raise HTTPException(status_code=503, detail=_format_search_error(exc)) from exc

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
        return embedding_settings_view(cfg)

    @router.put("/settings/embedding")
    def update_embedding_settings(
        body: EmbeddingSettingsUpdate, request: Request
    ) -> dict[str, Any]:
        cfg = request.app.state.config
        if not cfg.config_path or not cfg.config_path.is_file():
            raise HTTPException(
                status_code=400,
                detail="Config file not found; cannot save embedding settings",
            )
        updates = body.model_dump(exclude_unset=True)
        if not updates:
            raise HTTPException(status_code=400, detail="No settings to update")
        try:
            new_cfg, model_adjustment = save_embedding_settings(cfg, updates)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except OSError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        _apply_config(request, new_cfg)
        result = embedding_settings_view(new_cfg)
        result["saved"] = True
        if model_adjustment:
            result["model_adjustment"] = model_adjustment
        if result.get("embedding_mismatch"):
            result["rebuild_required"] = True
        return result

    @router.post("/settings/embedding/test")
    def embedding_test(body: EmbeddingTestRequest, request: Request) -> dict[str, Any]:
        cfg = request.app.state.config
        try:
            backend = create_embedding_backend(cfg.embedding)
            vector = backend.embed_query(body.text)
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail=_format_search_error(exc),
            ) from exc
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
