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

EMBEDDING_PROVIDER_CHOICES = (
    "sentence_transformers",
    "openai_compatible",
    "ollama",
)

DEFAULT_EMBEDDING_MODELS = {
    "sentence_transformers": "intfloat/multilingual-e5-base",
    "openai_compatible": "text-embedding-3-small",
    "ollama": "nomic-embed-text",
}

OPENAI_EMBEDDING_MODELS = frozenset(
    {
        "text-embedding-3-small",
        "text-embedding-3-large",
        "text-embedding-ada-002",
    }
)

LOCAL_EMBEDDING_MODELS = frozenset(
    {
        "intfloat/multilingual-e5-small",
        "intfloat/multilingual-e5-base",
        "intfloat/multilingual-e5-large",
    }
)


def default_embedding_model(provider: str) -> str:
    normalized = normalize_embedding_provider(provider)
    return DEFAULT_EMBEDDING_MODELS.get(
        normalized, DEFAULT_EMBEDDING_MODELS["sentence_transformers"]
    )


def coerce_embedding_model(
    provider: str,
    model: str,
    *,
    previous_provider: str | None = None,
) -> tuple[str, str | None]:
    """Return model for provider; second value is a user-facing adjustment note."""
    normalized = normalize_embedding_provider(provider)
    cleaned = model.strip()
    if not cleaned:
        resolved = default_embedding_model(normalized)
        return resolved, None

    if normalized == "sentence_transformers" and cleaned in OPENAI_EMBEDDING_MODELS:
        resolved = default_embedding_model(normalized)
        return resolved, (
            f"Модель {cleaned!r} — для OpenAI API; для локального провайдера "
            f"используется {resolved!r}"
        )

    if (
        normalized in {"openai_compatible", "ollama"}
        and cleaned in LOCAL_EMBEDDING_MODELS
        and previous_provider in {None, "sentence_transformers", "huggingface"}
    ):
        resolved = default_embedding_model(normalized)
        return resolved, (
            f"Модель {cleaned!r} — для локального E5; для {normalized} используется {resolved!r}"
        )

    return cleaned, None


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
        self.hbk_dir = _resolve_config_path(root, self.hbk_dir)
        self.data_dir = _resolve_config_path(root, self.data_dir)
        self.export_dir = _resolve_config_path(root, self.export_dir)
        self.index_dir = _resolve_config_path(root, self.index_dir)
        self.benchmark_results_path = _resolve_config_path(root, self.benchmark_results_path)
        if self.java_exporter.jar_path:
            jar = Path(self.java_exporter.jar_path)
            self.java_exporter.jar_path = str(
                _resolve_config_path(root, jar) if not jar.is_absolute() else jar.resolve()
            )
        if self.bsp.path:
            bsp = Path(self.bsp.path)
            self.bsp.path = str(
                _resolve_config_path(root, bsp) if not bsp.is_absolute() else bsp.resolve()
            )
        if self.mcp.log_file:
            log = Path(self.mcp.log_file)
            self.mcp.log_file = str(
                _resolve_config_path(root, log) if not log.is_absolute() else log.resolve()
            )


def _resolve_config_path(root: Path, value: Path) -> Path:
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


def _load_search_config(search_raw: dict[str, Any]) -> SearchConfig:
    return SearchConfig(
        dense_top_k=int(search_raw.get("dense_top_k", 20)),
        bm25_top_k=int(search_raw.get("bm25_top_k", 20)),
        final_top_k=int(search_raw.get("final_top_k", 5)),
        rrf_k=int(search_raw.get("rrf_k", 60)),
        build_vector_index=bool(search_raw.get("build_vector_index", True)),
    )


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
            model=str(embedding_raw.get("model", "intfloat/multilingual-e5-base")),
            device=str(embedding_raw.get("device", "cpu")),
            base_url=str(embedding_raw.get("base_url", "")),
            api_key=str(embedding_raw.get("api_key", "")),
            api_key_env=str(embedding_raw.get("api_key_env", "OPENAI_API_KEY")),
            batch_size=int(embedding_raw.get("batch_size", 64)),
            timeout=float(embedding_raw.get("timeout", 120.0)),
            query_prefix=str(embedding_raw.get("query_prefix", "query: ")),
            passage_prefix=str(embedding_raw.get("passage_prefix", "passage: ")),
        ),
        search=_load_search_config(search_raw),
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


def align_embedding_with_index(cfg: AppConfig) -> EmbeddingConfig:
    """Use index build metadata for query embeddings when config.yaml differs."""
    from dataclasses import replace

    from sntx_sem.index.meta import load_index_meta

    meta = load_index_meta(cfg.index_dir)
    indexed_count = int(meta.get("indexed_count", 0))
    if indexed_count <= 0 or not index_embedding_mismatch(cfg, meta, indexed_count=indexed_count):
        return cfg.embedding

    built_provider = normalize_embedding_provider(str(meta.get("embedding_provider", "")))
    built_model = str(meta.get("embedding_model", ""))
    if not built_model:
        return cfg.embedding

    aligned = replace(
        cfg.embedding,
        provider=built_provider,
        model=built_model,
    )
    if built_model.startswith("intfloat/multilingual-e5"):
        aligned = replace(
            aligned,
            query_prefix="query: ",
            passage_prefix="passage: ",
        )
    return aligned


def embedding_settings_view(cfg: AppConfig) -> dict[str, Any]:
    """Non-secret embedding settings for API and Web-UI."""
    emb = cfg.embedding
    db_status = bundled_database_status(cfg)
    index_info = db_status.get("index", {})
    return {
        "provider": resolve_embedding_provider(emb),
        "model": emb.model,
        "device": emb.device,
        "base_url": emb.base_url or "",
        "api_key_set": bool(emb.resolved_api_key),
        "api_key_env": emb.api_key_env,
        "query_prefix": emb.query_prefix,
        "passage_prefix": emb.passage_prefix,
        "embedding_mismatch": index_info.get("embedding_mismatch", False),
        "index_embedding_model": index_info.get("embedding_model"),
        "index_embedding_provider": index_info.get("embedding_provider"),
        "providers": list(EMBEDDING_PROVIDER_CHOICES),
        "default_models": dict(DEFAULT_EMBEDDING_MODELS),
        "config_writable": bool(cfg.config_path and cfg.config_path.is_file()),
    }


def save_embedding_settings(
    cfg: AppConfig, updates: dict[str, Any]
) -> tuple[AppConfig, str | None]:
    """Update embedding section in config.yaml and reload config."""
    if cfg.config_path is None or not cfg.config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {cfg.config_path}")

    with cfg.config_path.open(encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}

    emb_raw: dict[str, Any] = dict(raw.get("embedding") or {})
    previous_provider = resolve_embedding_provider(cfg.embedding)
    model_adjustment: str | None = None

    provider = updates.get("provider")
    if provider is not None:
        if provider not in EMBEDDING_PROVIDER_CHOICES:
            raise ValueError(f"Unknown embedding provider: {provider}")
        emb_raw["provider"] = provider

    next_provider = normalize_embedding_provider(str(emb_raw.get("provider") or previous_provider))
    if "model" in updates and updates["model"] is not None:
        requested_model = str(updates["model"])
    else:
        requested_model = str(emb_raw.get("model") or cfg.embedding.model)

    resolved_model, model_adjustment = coerce_embedding_model(
        next_provider,
        requested_model,
        previous_provider=previous_provider if provider is not None else None,
    )
    emb_raw["model"] = resolved_model

    for key in ("device", "base_url", "api_key_env", "query_prefix", "passage_prefix"):
        if key in updates and updates[key] is not None:
            emb_raw[key] = updates[key]

    api_key = updates.get("api_key")
    if api_key:
        emb_raw["api_key"] = api_key

    raw["embedding"] = emb_raw

    with cfg.config_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(raw, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    return load_config(cfg.config_path), model_adjustment


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
        "bsp": {
            "path": cfg.bsp.path or None,
            "enabled": cfg.bsp.enabled,
        },
    }


def count_jsonl_lines(path: Path) -> int:
    if not path.is_file():
        return 0
    count = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def count_domain_in_jsonl(path: Path, domain: str) -> int:
    if not path.is_file():
        return 0
    marker = f'"domain": "{domain}"'
    count = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if marker in line:
                count += 1
    return count


def count_domain_in_meta(meta_file: Path, domain: str) -> int:
    if not meta_file.is_file():
        return 0
    marker = f'"domain": "{domain}"'
    count = 0
    with meta_file.open(encoding="utf-8") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            count += block.count(marker)
    return count


def estimate_indexed_chunks(index_dir: Path, index_meta: dict[str, Any]) -> int:
    """Estimate chunk count without loading large chunks_meta.json into memory."""
    built_count = index_meta.get("indexed_count")
    if built_count is not None:
        return int(built_count)

    meta_file = index_dir / "chunks_meta.json"
    if not meta_file.is_file():
        return 0

    count = 0
    with meta_file.open(encoding="utf-8") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            count += block.count('"id":')
    return count


def collect_database_issues(
    *,
    ready: bool,
    chunks_file: Path,
    meta_file: Path,
    lance_dir: Path,
    export_count: int,
    indexed_count: int,
    embedding_mismatch: bool,
    partial_index: bool,
    config_model: str,
    index_model: str | None,
    build_vector_index: bool = True,
    vector_index_built: bool | None = None,
    lance_corrupted: bool = False,
    bsp_expected: bool = False,
    bsp_indexed_count: int = 0,
) -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []

    if not chunks_file.is_file():
        issues.append(
            {
                "severity": "error",
                "code": "export_missing",
                "message": ("Нет export/all_chunks.jsonl — выполните Ingest HBK + Index в /admin."),
            }
        )

    if export_count > 0 and not lance_dir.is_dir():
        issues.append(
            {
                "severity": "error",
                "code": "index_missing",
                "message": (
                    "Векторный индекс отсутствует или rebuild прервался. "
                    "Выполните Rebuild Index в /admin."
                ),
            }
        )

    if meta_file.is_file() and not lance_dir.is_dir():
        issues.append(
            {
                "severity": "error",
                "code": "index_broken",
                "message": (
                    "Метаданные индекса есть, но LanceDB (help_chunks.lance) удалён. "
                    "Поиск недоступен — нужен Rebuild Index."
                ),
            }
        )

    if lance_corrupted:
        issues.append(
            {
                "severity": "error",
                "code": "index_broken",
                "message": (
                    "Индекс LanceDB повреждён (неполный rebuild или отсутствуют файлы данных). "
                    "Выполните Rebuild Index в /admin. Для Docker ~4 ГБ RAM: "
                    "search.build_vector_index: false в config.yaml."
                ),
            }
        )

    if partial_index:
        issues.append(
            {
                "severity": "warning",
                "code": "partial_index",
                "message": (
                    f"Индекс неполный: {indexed_count} из {export_count} чанков. "
                    "Повторите Rebuild Index."
                ),
            }
        )

    if embedding_mismatch:
        issues.append(
            {
                "severity": "warning",
                "code": "embedding_mismatch",
                "message": (
                    f"Модель в config ({config_model}) не совпадает с индексом "
                    f"({index_model or 'неизвестно'}). Выполните Rebuild Index."
                ),
            }
        )

    if (
        ready
        and build_vector_index
        and indexed_count >= 256
        and lance_dir.is_dir()
        and vector_index_built is False
    ):
        issues.append(
            {
                "severity": "warning",
                "code": "vector_index_missing",
                "message": (
                    "IVF-индекс LanceDB не построен — поиск работает, но первый запрос может быть "
                    "медленнее. Увеличьте RAM и Rebuild Index или задайте "
                    "search.build_vector_index: false в config."
                ),
            }
        )

    if ready and bsp_expected and bsp_indexed_count == 0:
        issues.append(
            {
                "severity": "warning",
                "code": "bsp_not_indexed",
                "message": (
                    "BSP не проиндексирован — в индексе нет чанков домена bsp. "
                    "Выполните Ingest BSP + Rebuild Index в /admin."
                ),
            }
        )

    if not ready and not any(
        item["code"] in {"export_missing", "index_missing", "index_broken"} for item in issues
    ):
        issues.append(
            {
                "severity": "error",
                "code": "not_ready",
                "message": "База не готова к поиску. Проверьте ingest и rebuild в /admin.",
            }
        )

    return issues


def bundled_database_status(cfg: AppConfig) -> dict[str, Any]:
    """Check whether the local help database is ready."""
    from sntx_sem.index.integrity import check_lance_index
    from sntx_sem.index.meta import load_index_meta

    index_meta = load_index_meta(cfg.index_dir)
    chunks_file = cfg.export_dir / "all_chunks.jsonl"
    meta_file = cfg.index_dir / "chunks_meta.json"
    lance_dir = cfg.index_dir / "help_chunks.lance"

    lance_ok = True
    if lance_dir.is_dir():
        lance_ok, _lance_error = check_lance_index(cfg.index_dir)
    lance_corrupted = lance_dir.is_dir() and not lance_ok

    indexed_count = estimate_indexed_chunks(cfg.index_dir, index_meta)
    export_count = count_jsonl_lines(chunks_file)

    ready = (
        chunks_file.is_file()
        and meta_file.is_file()
        and lance_dir.is_dir()
        and lance_ok
        and indexed_count > 0
        and (not export_count or indexed_count >= export_count * 0.95)
    )

    embedding_mismatch = False
    if indexed_count > 0 and not index_meta:
        embedding_mismatch = True
    elif indexed_count > 0:
        embedding_mismatch = index_embedding_mismatch(cfg, index_meta, indexed_count=indexed_count)

    partial_index = indexed_count > 0 and export_count > 0 and indexed_count < export_count * 0.95
    config_model = cfg.embedding.model
    index_model = index_meta.get("embedding_model") if index_meta else None
    vector_index_built = index_meta.get("vector_index_built") if index_meta else None
    if vector_index_built is not None:
        vector_index_built = bool(vector_index_built)
    bsp_expected = bool(cfg.bsp.path) or cfg.bsp.enabled
    bsp_indexed_count = count_domain_in_meta(meta_file, "bsp") if meta_file.is_file() else 0
    issues = collect_database_issues(
        ready=ready,
        chunks_file=chunks_file,
        meta_file=meta_file,
        lance_dir=lance_dir,
        export_count=export_count,
        indexed_count=indexed_count,
        embedding_mismatch=embedding_mismatch,
        partial_index=partial_index,
        config_model=config_model,
        index_model=str(index_model) if index_model else None,
        build_vector_index=cfg.search.build_vector_index,
        vector_index_built=vector_index_built,
        lance_corrupted=lance_corrupted,
        bsp_expected=bsp_expected,
        bsp_indexed_count=bsp_indexed_count,
    )

    return {
        "ready": ready,
        "issues": issues,
        "config": config_summary(cfg),
        "index": {
            "platform_version": index_meta.get("platform_version", cfg.platform_version),
            "embedding_provider": index_meta.get("embedding_provider"),
            "embedding_model": index_model,
            "embedding_dimensions": index_meta.get("embedding_dimensions"),
            "embedding_mismatch": embedding_mismatch,
            "export_chunks": export_count,
            "indexed_chunks": indexed_count,
            "built_at": index_meta.get("built_at"),
            "partial_index": partial_index,
            "lance_present": lance_dir.is_dir(),
            "lance_ok": lance_ok,
            "vector_index_built": vector_index_built,
        },
        "embedding_in_sync": not embedding_mismatch,
        "paths": {
            "chunks_file": str(chunks_file),
            "index_dir": str(cfg.index_dir),
            "build_meta": str(cfg.index_dir / "build_meta.json"),
        },
    }
