"""Configuration dataclasses and path resolution."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


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
    model: str = "intfloat/multilingual-e5-base"
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


@dataclass
class SearchConfig:
    dense_top_k: int = 20
    bm25_top_k: int = 20
    final_top_k: int = 5
    rrf_k: int = 60
    build_vector_index: bool = True


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
        self.hbk_dir = resolve_config_path(root, self.hbk_dir)
        self.data_dir = resolve_config_path(root, self.data_dir)
        self.export_dir = resolve_config_path(root, self.export_dir)
        self.index_dir = resolve_config_path(root, self.index_dir)
        self.benchmark_results_path = resolve_config_path(root, self.benchmark_results_path)
        if self.java_exporter.jar_path:
            jar = Path(self.java_exporter.jar_path)
            self.java_exporter.jar_path = str(
                resolve_config_path(root, jar) if not jar.is_absolute() else jar.resolve()
            )
        if self.bsp.path:
            bsp = Path(self.bsp.path)
            self.bsp.path = str(
                resolve_config_path(root, bsp) if not bsp.is_absolute() else bsp.resolve()
            )
        if self.mcp.log_file:
            log = Path(self.mcp.log_file)
            self.mcp.log_file = str(
                resolve_config_path(root, log) if not log.is_absolute() else log.resolve()
            )


def resolve_config_path(root: Path, value: Path) -> Path:
    if value.is_absolute():
        return value.resolve()
    return (root / value).resolve()


REQUIRED_HBK = [
    "shcntx_ru.hbk",
    "shcntx_root.hbk",
    "shlang_ru.hbk",
    "shlang_root.hbk",
    "shquery_ru.hbk",
    "shquery_root.hbk",
]

OPTIONAL_HBK = REQUIRED_HBK
