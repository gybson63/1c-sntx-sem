"""Tests for background job progress reporting."""

from __future__ import annotations

import errno

from sntx_sem.api.errors import format_job_error
from sntx_sem.api.jobs import Job, JobProgress, JobStatus, JobType


def test_job_detail_includes_phase_and_progress() -> None:
    job = Job(
        id="test-id",
        type=JobType.INDEX,
        status=JobStatus.RUNNING,
        created_at="2026-06-27T00:00:00+00:00",
        progress=JobProgress(
            phase="embeddings",
            label="Эмбеддинги",
            current=512,
            total=30524,
        ),
    )
    detail = job.to_detail()
    assert detail["phase"] == "embeddings"
    assert detail["phase_label"] == "Эмбеддинги"
    assert detail["progress"] == {"current": 512, "total": 30524, "percent": 2}


def test_format_job_error_enomem() -> None:
    message = format_job_error(OSError(errno.ENOMEM, "Cannot allocate memory"))
    assert "памят" in message.lower()


def test_format_job_error_file_not_found() -> None:
    message = format_job_error(FileNotFoundError("/tmp/missing.hbk"))
    assert "не найден" in message.lower()
