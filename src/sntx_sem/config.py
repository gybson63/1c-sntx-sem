"""Application configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class LocalConfig:
    path: str
    label: str


@dataclass
class LLMConfig:
    provider: str = "openai_compatible"
    base_url: str = ""
    api_key_env: str = "OPENAI_API_KEY"
    model: str = "auto"

    @property
    def api_key(self) -> str | None:
        return os.environ.get(self.api_key_env)


@dataclass
class EmbeddingConfig:
    model: str = "intfloat/multilingual-e5-small"
    device: str = "cpu"


@dataclass
class SearchConfig:
    dense_top_k: int = 20
    bm25_top_k: int = 20
    final_top_k: int = 5
    rrf_k: int = 60


@dataclass
class JavaExporterConfig:
    jar_path: str = ""
    enabled: bool = True


@dataclass
class AppConfig:
    platform_version: str = "8.3.27"
    hbk_dir: Path = field(default_factory=lambda: Path("./hbk"))
    data_dir: Path = field(default_factory=lambda: Path("./data"))
    export_dir: Path = field(default_factory=lambda: Path("./data/export"))
    index_dir: Path = field(default_factory=lambda: Path("./data/index"))
    local_configs: list[LocalConfig] = field(default_factory=list)
    llm: LLMConfig = field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    java_exporter: JavaExporterConfig = field(default_factory=JavaExporterConfig)
    benchmark_results_path: Path = field(
        default_factory=lambda: Path("./config/benchmark_results.yaml")
    )

    def resolve_paths(self, base: Path | None = None) -> None:
        root = base or Path.cwd()
        self.hbk_dir = (root / self.hbk_dir).resolve()
        self.data_dir = (root / self.data_dir).resolve()
        self.export_dir = (root / self.export_dir).resolve()
        self.index_dir = (root / self.index_dir).resolve()
        self.benchmark_results_path = (root / self.benchmark_results_path).resolve()
        if self.java_exporter.jar_path:
            self.java_exporter.jar_path = str(
                (root / self.java_exporter.jar_path).resolve()
            )


REQUIRED_HBK = [
    "shcntx_ru.hbk",
    "shcntx_root.hbk",
    "shlang_ru.hbk",
    "shlang_root.hbk",
    "shquery_ru.hbk",
    "shquery_root.hbk",
]

OPTIONAL_HBK = REQUIRED_HBK


def load_config(path: str | Path | None = None) -> AppConfig:
    config_path = Path(
        path or os.environ.get("SNTX_SEM_CONFIG", "config.yaml")
    )
    if not config_path.is_file():
        cfg = AppConfig()
        cfg.resolve_paths(config_path.parent if config_path.name != "config.yaml" else Path.cwd())
        return cfg

    with config_path.open(encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}

    local_configs = [
        LocalConfig(path=str(item["path"]), label=str(item["label"]))
        for item in raw.get("local_configs", [])
    ]

    llm_raw = raw.get("llm", {})
    embedding_raw = raw.get("embedding", {})
    search_raw = raw.get("search", {})
    java_raw = raw.get("java_exporter", {})

    cfg = AppConfig(
        platform_version=str(raw.get("platform_version", "8.3.27")),
        hbk_dir=Path(raw.get("hbk_dir", "./hbk")),
        data_dir=Path(raw.get("data_dir", "./data")),
        export_dir=Path(raw.get("export_dir", "./data/export")),
        index_dir=Path(raw.get("index_dir", "./data/index")),
        local_configs=local_configs,
        llm=LLMConfig(
            provider=str(llm_raw.get("provider", "openai_compatible")),
            base_url=str(llm_raw.get("base_url", "")),
            api_key_env=str(llm_raw.get("api_key_env", "OPENAI_API_KEY")),
            model=str(llm_raw.get("model", "auto")),
        ),
        embedding=EmbeddingConfig(
            model=str(embedding_raw.get("model", "intfloat/multilingual-e5-small")),
            device=str(embedding_raw.get("device", "cpu")),
        ),
        search=SearchConfig(
            dense_top_k=int(search_raw.get("dense_top_k", 20)),
            bm25_top_k=int(search_raw.get("bm25_top_k", 20)),
            final_top_k=int(search_raw.get("final_top_k", 5)),
            rrf_k=int(search_raw.get("rrf_k", 60)),
        ),
        java_exporter=JavaExporterConfig(
            jar_path=str(java_raw.get("jar_path", "")),
            enabled=bool(java_raw.get("enabled", True)),
        ),
    )
    cfg.resolve_paths(config_path.parent)
    return cfg


def list_hbk_files(hbk_dir: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    if not hbk_dir.is_dir():
        return result
    for path in hbk_dir.glob("*.hbk"):
        result[path.name] = path
    return result


def detect_platform_path() -> Path | None:
    candidates = [
        Path(r"C:\Program Files\1cv8"),
        Path("/opt/1cv8"),
        Path("/opt/1C/v8"),
    ]
    for base in candidates:
        if not base.is_dir():
            continue
        for pattern in ["**/bin/shcntx_ru.hbk", "**/shcntx_ru.hbk"]:
            matches = sorted(base.glob(pattern), reverse=True)
            if matches:
                return matches[0].parent if matches[0].name == "shcntx_ru.hbk" else matches[0].parent.parent
    return None


def load_manifest(data_dir: Path) -> dict[str, Any]:
    path = data_dir / "manifest.yaml"
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def bundled_database_status(cfg: AppConfig) -> dict[str, Any]:
    """Check whether the pre-built database shipped with the repo is ready."""
    manifest = load_manifest(cfg.data_dir)
    chunks_file = cfg.export_dir / "all_chunks.jsonl"
    meta_file = cfg.index_dir / "chunks_meta.json"
    lance_dir = cfg.index_dir / "help_chunks.lance"

    indexed_count = 0
    if meta_file.is_file():
        import json

        indexed_count = len(json.loads(meta_file.read_text(encoding="utf-8")))

    export_count = manifest.get("chunk_count", 0)
    if chunks_file.is_file() and not export_count:
        export_count = sum(1 for _ in chunks_file.open(encoding="utf-8") if _.strip())

    ready = (
        chunks_file.is_file()
        and meta_file.is_file()
        and lance_dir.is_dir()
        and indexed_count > 0
        and (not export_count or indexed_count >= export_count * 0.95)
    )

    return {
        "ready": ready,
        "platform_version": manifest.get("platform_version", cfg.platform_version),
        "embedding_model": manifest.get("embedding_model", cfg.embedding.model),
        "export_chunks": export_count,
        "indexed_chunks": indexed_count,
        "chunks_file": str(chunks_file),
        "index_dir": str(cfg.index_dir),
        "built_at": manifest.get("built_at"),
        "partial_index": indexed_count > 0 and export_count and indexed_count < export_count * 0.95,
    }


def update_manifest_indexed_count(data_dir: Path, indexed_count: int) -> None:
    """Update manifest after index rebuild."""
    from datetime import date

    path = data_dir / "manifest.yaml"
    manifest = load_manifest(data_dir)
    manifest["indexed_count"] = indexed_count
    manifest["built_at"] = date.today().isoformat()
    with path.open("w", encoding="utf-8") as f:
        yaml.dump(manifest, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
