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
class Job:
    id: str
    type: JobType
    status: JobStatus
    created_at: str
    logs: list[str] = field(default_factory=list)
    result: dict[str, Any] | None = None
    error: str | None = None

    def to_detail(self, *, since_log: int = 0) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.value,
            "status": self.status.value,
            "created_at": self.created_at,
            "logs": self.logs[since_log:],
            "log_offset": len(self.logs),
            "result": self.result,
            "error": self.error,
        }


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
        logs: list[str] = []

        def log_cb(line: str) -> None:
            logs.append(line)
            self._append_log(job_id, line)

        try:
            stats = run_ingest_hbk(cfg, cfg.hbk_dir, cfg.platform_version, log=logs)
            count = build_index(cfg, rebuild=True, log=logs)
            for line in logs:
                self._append_log(job_id, line)
            self._finish(job_id, result={"ingest_stats": stats, "indexed_chunks": count})
        except Exception as exc:
            self._append_log(job_id, traceback.format_exc())
            self._finish(job_id, error=str(exc))

    def _run_ingest_bsp(self, job_id: str, cfg: AppConfig, bsp_dir: Path) -> None:
        self._set_status(job_id, JobStatus.RUNNING)
        logs: list[str] = []
        try:
            stats = run_ingest_bsp(cfg, bsp_dir, log=logs)
            count = build_index(cfg, rebuild=True, log=logs)
            for line in logs:
                self._append_log(job_id, line)
            self._finish(job_id, result={"bsp_stats": stats, "indexed_chunks": count})
        except Exception as exc:
            self._append_log(job_id, traceback.format_exc())
            self._finish(job_id, error=str(exc))

    def _run_index(self, job_id: str, cfg: AppConfig, rebuild: bool) -> None:
        self._set_status(job_id, JobStatus.RUNNING)
        logs: list[str] = []

        try:
            count = build_index(cfg, rebuild=rebuild, log=logs)
            for line in logs:
                self._append_log(job_id, line)
            self._finish(job_id, result={"indexed_chunks": count})
        except Exception as exc:
            self._append_log(job_id, traceback.format_exc())
            self._finish(job_id, error=str(exc))
