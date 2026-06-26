"""Application configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


def resolve_api_key(api_key: str, api_key_env: str) -> str | None:
    """Return API key from config value or named environment variable."""
    if api_key:
        return api_key
    if api_key_env:
        return os.environ.get(api_key_env)
    return None


@dataclass
class LocalConfig:
    path: str
    label: str


@dataclass
class LLMConfig:
    provider: str = "openai_compatible"
    base_url: str = ""
    api_key: str = ""
    api_key_env: str = "OPENAI_API_KEY"
    model: str = "auto"

    @property
    def resolved_api_key(self) -> str | None:
        return resolve_api_key(self.api_key, self.api_key_env)


@dataclass
class EmbeddingConfig:
    provider: str = ""
    model: str = "intfloat/multilingual-e5-small"
    device: str = "cpu"
    base_url: str = ""
    api_key: str = ""
    api_key_env: str = "OPENAI_API_KEY"
    batch_size: int = 64
    timeout: float = 120.0
    query_prefix: str = "query: "
    passage_prefix: str = "passage: "

    @property
    def resolved_api_key(self) -> str | None:
        return resolve_api_key(self.api_key, self.api_key_env)


def resolve_embedding_provider(cfg: EmbeddingConfig) -> str:
    """Resolve embedding provider: explicit > base_url > sentence_transformers."""
    if cfg.provider:
        if cfg.provider == "huggingface":
            return "sentence_transformers"
        return cfg.provider
    if cfg.base_url:
        return "openai_compatible"
    return "sentence_transformers"


LOCAL_EMBEDDING_PROVIDERS = frozenset({"sentence_transformers", "huggingface"})


def normalize_embedding_provider(provider: str) -> str:
    """Map legacy huggingface alias to sentence_transformers."""
    if provider == "huggingface":
        return "sentence_transformers"
    return provider


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
class BspConfig:
    path: str = ""
    enabled: bool = False


@dataclass
class McpConfig:
    log_level: str = "INFO"
    log_file: str = ""
    log_max_chars: int = 2000


@dataclass
class ApiConfig:
    host: str = "127.0.0.1"
    port: int = 8000


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
    bsp: BspConfig = field(default_factory=BspConfig)
    mcp: McpConfig = field(default_factory=McpConfig)
    api: ApiConfig = field(default_factory=ApiConfig)
    benchmark_results_path: Path = field(
        default_factory=lambda: Path("./config/benchmark_results.yaml")
    )
    config_path: Path | None = None

    def resolve_paths(self, base: Path | None = None) -> None:
        root = base or Path.cwd()
        self.hbk_dir = (root / self.hbk_dir).resolve()
        self.data_dir = (root / self.data_dir).resolve()
        self.export_dir = (root / self.export_dir).resolve()
        self.index_dir = (root / self.index_dir).resolve()
        self.benchmark_results_path = (root / self.benchmark_results_path).resolve()
        if self.java_exporter.jar_path:
            self.java_exporter.jar_path = str((root / self.java_exporter.jar_path).resolve())
        if self.bsp.path:
            self.bsp.path = str((root / self.bsp.path).resolve())
        if self.mcp.log_file:
            self.mcp.log_file = str((root / self.mcp.log_file).resolve())


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
    config_path = Path(path or os.environ.get("SNTX_SEM_CONFIG", "config.yaml"))
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
    bsp_raw = raw.get("bsp", {})
    mcp_raw = raw.get("mcp", {})
    api_raw = raw.get("api", {})

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
            api_key=str(llm_raw.get("api_key", "")),
            api_key_env=str(llm_raw.get("api_key_env", "OPENAI_API_KEY")),
            model=str(llm_raw.get("model", "auto")),
        ),
        embedding=EmbeddingConfig(
            provider=str(embedding_raw.get("provider", "")),
            model=str(embedding_raw.get("model", "intfloat/multilingual-e5-small")),
            device=str(embedding_raw.get("device", "cpu")),
            base_url=str(embedding_raw.get("base_url", "")),
            api_key=str(embedding_raw.get("api_key", "")),
            api_key_env=str(embedding_raw.get("api_key_env", "OPENAI_API_KEY")),
            batch_size=int(embedding_raw.get("batch_size", 64)),
            timeout=float(embedding_raw.get("timeout", 120.0)),
            query_prefix=str(embedding_raw.get("query_prefix", "query: ")),
            passage_prefix=str(embedding_raw.get("passage_prefix", "passage: ")),
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
        bsp=BspConfig(
            path=str(bsp_raw.get("path", "")),
            enabled=bool(bsp_raw.get("enabled", False)),
        ),
        mcp=McpConfig(
            log_level=str(mcp_raw.get("log_level", "INFO")),
            log_file=str(mcp_raw.get("log_file", "")),
            log_max_chars=int(mcp_raw.get("log_max_chars", 2000)),
        ),
        api=ApiConfig(
            host=str(api_raw.get("host", "127.0.0.1")),
            port=int(api_raw.get("port", 8000)),
        ),
    )
    cfg.config_path = config_path.resolve()
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
                return (
                    matches[0].parent
                    if matches[0].name == "shcntx_ru.hbk"
                    else matches[0].parent.parent
                )
    return None


def index_embedding_mismatch(
    cfg: AppConfig,
    index_meta: dict[str, Any],
    *,
    indexed_count: int,
) -> bool:
    """True when config embedding settings differ from the built index metadata."""
    if indexed_count <= 0:
        return False

    current_provider = normalize_embedding_provider(resolve_embedding_provider(cfg.embedding))
    current_model = cfg.embedding.model
    built_provider = index_meta.get("embedding_provider")
    built_model = index_meta.get("embedding_model")

    if (
        built_provider is not None
        and normalize_embedding_provider(str(built_provider)) != current_provider
    ):
        return True
    return built_model is not None and built_model != current_model


def config_summary(cfg: AppConfig) -> dict[str, Any]:
    """Non-secret view of active settings from config.yaml."""
    embedding_provider = resolve_embedding_provider(cfg.embedding)
    return {
        "config_file": str(cfg.config_path) if cfg.config_path else None,
        "embedding": {
            "provider": embedding_provider,
            "model": cfg.embedding.model,
            "base_url": cfg.embedding.base_url or None,
            "api_key_set": bool(cfg.embedding.resolved_api_key),
        },
        "llm": {
            "provider": cfg.llm.provider,
            "model": cfg.llm.model,
            "base_url": cfg.llm.base_url or None,
            "api_key_set": bool(cfg.llm.resolved_api_key),
        },
        "mcp": {
            "log_level": cfg.mcp.log_level,
            "log_file": cfg.mcp.log_file or None,
        },
    }


def bundled_database_status(cfg: AppConfig) -> dict[str, Any]:
    """Check whether the local help database is ready."""
    from sntx_sem.index.meta import load_index_meta

    index_meta = load_index_meta(cfg.index_dir)
    chunks_file = cfg.export_dir / "all_chunks.jsonl"
    meta_file = cfg.index_dir / "chunks_meta.json"
    lance_dir = cfg.index_dir / "help_chunks.lance"

    indexed_count = 0
    if meta_file.is_file():
        import json

        indexed_count = len(json.loads(meta_file.read_text(encoding="utf-8")))

    export_count = 0
    if chunks_file.is_file():
        export_count = sum(1 for _ in chunks_file.open(encoding="utf-8") if _.strip())

    ready = (
        chunks_file.is_file()
        and meta_file.is_file()
        and lance_dir.is_dir()
        and indexed_count > 0
        and (not export_count or indexed_count >= export_count * 0.95)
    )

    embedding_mismatch = False
    if indexed_count > 0 and not index_meta:
        embedding_mismatch = True
    elif indexed_count > 0:
        embedding_mismatch = index_embedding_mismatch(cfg, index_meta, indexed_count=indexed_count)

    return {
        "ready": ready,
        "config": config_summary(cfg),
        "index": {
            "platform_version": index_meta.get("platform_version", cfg.platform_version),
            "embedding_provider": index_meta.get("embedding_provider"),
            "embedding_model": index_meta.get("embedding_model"),
            "embedding_dimensions": index_meta.get("embedding_dimensions"),
            "embedding_mismatch": embedding_mismatch,
            "export_chunks": export_count,
            "indexed_chunks": indexed_count,
            "built_at": index_meta.get("built_at"),
            "partial_index": (
                indexed_count > 0 and export_count and indexed_count < export_count * 0.95
            ),
        },
        "embedding_in_sync": not embedding_mismatch,
        "paths": {
            "chunks_file": str(chunks_file),
            "index_dir": str(cfg.index_dir),
            "build_meta": str(cfg.index_dir / "build_meta.json"),
        },
    }
