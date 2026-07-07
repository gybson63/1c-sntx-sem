"""Background ingest/index jobs."""

from __future__ import annotations

import threading
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from sntx_sem.api.errors import format_job_error
from sntx_sem.config import AppConfig
from sntx_sem.indexing import build_index, run_ingest_bsp, run_ingest_hbk


class JobStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class JobType(StrEnum):
    INGEST = "ingest"
    INGEST_BSP = "ingest_bsp"
    INDEX = "index"


@dataclass
class JobProgress:
    phase: str = ""
    label: str = ""
    current: int = 0
    total: int = 0


@dataclass
class Job:
    id: str
    type: JobType
    status: JobStatus
    created_at: str
    logs: list[str] = field(default_factory=list)
    result: dict[str, Any] | None = None
    error: str | None = None
    progress: JobProgress = field(default_factory=JobProgress)

    def to_detail(self, *, since_log: int = 0) -> dict[str, Any]:
        detail: dict[str, Any] = {
            "id": self.id,
            "type": self.type.value,
            "status": self.status.value,
            "created_at": self.created_at,
            "phase": self.progress.phase,
            "phase_label": self.progress.label,
            "logs": self.logs[since_log:],
            "log_offset": len(self.logs),
            "result": self.result,
            "error": self.error,
        }
        if self.progress.total > 0:
            detail["progress"] = {
                "current": self.progress.current,
                "total": self.progress.total,
                "percent": round(100 * self.progress.current / self.progress.total),
            }
        return detail


class JobStore:
    MAX_JOBS = 50

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def start_ingest(self, cfg: AppConfig) -> Job:
        job = self._create(JobType.INGEST)
        threading.Thread(
            target=self._run_ingest,
            args=(job.id, cfg),
            name=f"sntx-sem-ingest-{job.id[:8]}",
            daemon=True,
        ).start()
        return job

    def start_ingest_bsp(self, cfg: AppConfig, bsp_dir: Path) -> Job:
        job = self._create(JobType.INGEST_BSP)
        threading.Thread(
            target=self._run_ingest_bsp,
            args=(job.id, cfg, bsp_dir),
            name=f"sntx-sem-ingest-bsp-{job.id[:8]}",
            daemon=True,
        ).start()
        return job

    def start_index(self, cfg: AppConfig, *, rebuild: bool = True) -> Job:
        job = self._create(JobType.INDEX)
        threading.Thread(
            target=self._run_index,
            args=(job.id, cfg, rebuild),
            name=f"sntx-sem-index-{job.id[:8]}",
            daemon=True,
        ).start()
        return job

    def _create(self, job_type: JobType) -> Job:
        job = Job(
            id=str(uuid.uuid4()),
            type=job_type,
            status=JobStatus.PENDING,
            created_at=datetime.now(UTC).isoformat(),
        )
        with self._lock:
            self._jobs[job.id] = job
            if len(self._jobs) > self.MAX_JOBS:
                oldest = min(self._jobs.values(), key=lambda j: j.created_at)
                self._jobs.pop(oldest.id, None)
        return job

    def _append_log(self, job_id: str, line: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.logs.append(line)

    def _set_status(self, job_id: str, status: JobStatus) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.status = status

    def _set_progress(
        self,
        job_id: str,
        *,
        phase: str,
        label: str,
        current: int = 0,
        total: int = 0,
    ) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.progress = JobProgress(
                    phase=phase,
                    label=label,
                    current=current,
                    total=total,
                )

    def _finish(
        self, job_id: str, *, result: dict[str, Any] | None = None, error: str | None = None
    ) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            job.status = JobStatus.FAILED if error else JobStatus.COMPLETED
            job.result = result
            job.error = error

    def _run_ingest(self, job_id: str, cfg: AppConfig) -> None:
        self._set_status(job_id, JobStatus.RUNNING)

        def log_line(line: str) -> None:
            self._append_log(job_id, line)

        def on_progress(current: int, total: int) -> None:
            self._set_progress(
                job_id,
                phase="embeddings",
                label="Эмбеддинги",
                current=current,
                total=total,
            )
            if total and (current == total or current % max(1, total // 20) == 0):
                pct = 100 * current // total
                log_line(f"Эмбеддинги: {current}/{total} ({pct}%)")

        try:
            self._set_progress(
                job_id,
                phase="ingest_hbk",
                label="Шаг 1/2: извлечение статей из HBK",
            )
            log_line("=== Шаг 1/2: извлечение статей из HBK ===")
            stats = run_ingest_hbk(cfg, cfg.hbk_dir, cfg.platform_version, log=log_line)
            self._set_progress(
                job_id,
                phase="build_index",
                label="Шаг 2/2: построение индекса",
            )
            log_line("=== Шаг 2/2: построение индекса (эмбеддинги) ===")
            count = build_index(cfg, rebuild=True, log=log_line, on_progress=on_progress)
            self._finish(job_id, result={"ingest_stats": stats, "indexed_chunks": count})
        except Exception as exc:
            self._append_log(job_id, traceback.format_exc())
            self._finish(job_id, error=format_job_error(exc, traceback_text=traceback.format_exc()))

    def _run_ingest_bsp(self, job_id: str, cfg: AppConfig, bsp_dir: Path) -> None:
        self._set_status(job_id, JobStatus.RUNNING)

        def log_line(line: str) -> None:
            self._append_log(job_id, line)

        def on_progress(current: int, total: int) -> None:
            self._set_progress(
                job_id,
                phase="embeddings",
                label="Эмбеддинги",
                current=current,
                total=total,
            )
            if total and (current == total or current % max(1, total // 20) == 0):
                pct = 100 * current // total
                log_line(f"Эмбеддинги: {current}/{total} ({pct}%)")

        try:
            self._set_progress(
                job_id,
                phase="ingest_bsp",
                label="Шаг 1/2: извлечение API БСП",
            )
            log_line("=== Шаг 1/2: извлечение API БСП ===")
            stats = run_ingest_bsp(cfg, bsp_dir, log=log_line)
            self._set_progress(
                job_id,
                phase="build_index",
                label="Шаг 2/2: построение индекса",
            )
            log_line("=== Шаг 2/2: построение индекса (эмбеддинги) ===")
            count = build_index(cfg, rebuild=True, log=log_line, on_progress=on_progress)
            self._finish(job_id, result={"bsp_stats": stats, "indexed_chunks": count})
        except Exception as exc:
            self._append_log(job_id, traceback.format_exc())
            self._finish(job_id, error=format_job_error(exc, traceback_text=traceback.format_exc()))

    def _run_index(self, job_id: str, cfg: AppConfig, rebuild: bool) -> None:
        self._set_status(job_id, JobStatus.RUNNING)

        def log_line(line: str) -> None:
            self._append_log(job_id, line)

        def on_progress(current: int, total: int) -> None:
            self._set_progress(
                job_id,
                phase="embeddings",
                label="Эмбеддинги",
                current=current,
                total=total,
            )
            if total and (current == total or current % max(1, total // 20) == 0):
                pct = 100 * current // total
                log_line(f"Эмбеддинги: {current}/{total} ({pct}%)")

        try:
            self._set_progress(job_id, phase="build_index", label="Построение индекса")
            log_line("=== Построение индекса (эмбеддинги) ===")
            count = build_index(cfg, rebuild=rebuild, log=log_line, on_progress=on_progress)
            self._finish(job_id, result={"indexed_chunks": count})
        except Exception as exc:
            self._append_log(job_id, traceback.format_exc())
            self._finish(job_id, error=format_job_error(exc, traceback_text=traceback.format_exc()))
