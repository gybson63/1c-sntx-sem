"""Application configuration — public API."""

from sntx_sem.config.diagnostics import (
    bundled_database_status,
    collect_database_issues,
    count_domain_in_jsonl,
    count_domain_in_meta,
    count_jsonl_lines,
    estimate_indexed_chunks,
    index_embedding_info,
    is_search_ready,
)
from sntx_sem.config.embedding import (
    DEFAULT_EMBEDDING_MODELS,
    EMBEDDING_PROVIDER_CHOICES,
    LOCAL_EMBEDDING_MODELS,
    LOCAL_EMBEDDING_PROVIDERS,
    OPENAI_EMBEDDING_MODELS,
    align_embedding_with_index,
    coerce_embedding_model,
    default_embedding_model,
    index_embedding_mismatch,
    normalize_embedding_provider,
    resolve_embedding_provider,
)
from sntx_sem.config.loader import detect_platform_path, list_hbk_files, load_config
from sntx_sem.config.models import (
    OPTIONAL_HBK,
    REQUIRED_HBK,
    ApiConfig,
    AppConfig,
    BspConfig,
    EmbeddingConfig,
    JavaExporterConfig,
    LLMConfig,
    LocalConfig,
    McpConfig,
    SearchConfig,
    resolve_api_key,
)
from sntx_sem.config.persistence import save_embedding_settings
from sntx_sem.config.status_cache import DatabaseStatusCache
from sntx_sem.config.summary import config_summary
from sntx_sem.config.validation import ConfigError
from sntx_sem.config.views import embedding_settings_view

__all__ = [
    "DEFAULT_EMBEDDING_MODELS",
    "EMBEDDING_PROVIDER_CHOICES",
    "LOCAL_EMBEDDING_MODELS",
    "LOCAL_EMBEDDING_PROVIDERS",
    "OPENAI_EMBEDDING_MODELS",
    "OPTIONAL_HBK",
    "REQUIRED_HBK",
    "ApiConfig",
    "AppConfig",
    "BspConfig",
    "ConfigError",
    "EmbeddingConfig",
    "JavaExporterConfig",
    "LLMConfig",
    "LocalConfig",
    "McpConfig",
    "SearchConfig",
    "align_embedding_with_index",
    "bundled_database_status",
    "coerce_embedding_model",
    "collect_database_issues",
    "config_summary",
    "count_domain_in_jsonl",
    "count_domain_in_meta",
    "count_jsonl_lines",
    "default_embedding_model",
    "detect_platform_path",
    "embedding_settings_view",
    "estimate_indexed_chunks",
    "DatabaseStatusCache",
    "index_embedding_info",
    "index_embedding_mismatch",
    "is_search_ready",
    "list_hbk_files",
    "load_config",
    "normalize_embedding_provider",
    "resolve_api_key",
    "resolve_embedding_provider",
    "save_embedding_settings",
]
