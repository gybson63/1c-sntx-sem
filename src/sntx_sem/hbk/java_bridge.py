"""Run Java bsl-context exporter when available."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from sntx_sem.hbk.extractor import HelpChunk, export_chunks_jsonl


def run_java_exporter(
    jar_path: Path,
    hbk_dir: Path,
    export_dir: Path,
    platform_version: str,
) -> list[HelpChunk] | None:
    if not jar_path.is_file():
        return None

    out_dir = export_dir / "java_bsl"
    out_dir.mkdir(parents=True, exist_ok=True)

    shcntx_ru = hbk_dir / "shcntx_ru.hbk"
    shcntx_root = hbk_dir / "shcntx_root.hbk"
    shlang_ru = hbk_dir / "shlang_ru.hbk"
    shlang_root = hbk_dir / "shlang_root.hbk"

    if not shcntx_ru.is_file():
        return None

    cmd = [
        "java",
        "-jar",
        str(jar_path),
        "--hbk-dir",
        str(hbk_dir),
        "--output",
        str(out_dir),
        "--platform-version",
        platform_version,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=600)
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return None

    jsonl_path = out_dir / "bsl_chunks.jsonl"
    if not jsonl_path.is_file():
        return None

    chunks: list[HelpChunk] = []
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            chunks.append(HelpChunk(**data))
    return chunks


def merge_java_export(chunks: list[HelpChunk], export_dir: Path) -> None:
    export_chunks_jsonl(chunks, export_dir / "bsl_java_chunks.jsonl")
