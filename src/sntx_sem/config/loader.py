"""Load and validate application configuration from YAML."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from sntx_sem.config.models import (
    ApiConfig,
    AppConfig,
    BspConfig,
    EmbeddingConfig,
    JavaExporterConfig,
    LLMConfig,
    LocalConfig,
    McpConfig,
    SearchConfig,
)
from sntx_sem.config.validation import (
    KNOWN_SECTION_KEYS,
    ConfigError,
    parse_bool,
    parse_path_value,
    parse_port,
    parse_positive_float,
    parse_positive_int,
    parse_string,
    require_mapping,
    validate_embedding_provider,
    validate_local_configs,
    validate_root,
    warn_unknown_keys,
)


def _load_search_config(search_raw: dict[str, Any], *, path: Path | None) -> SearchConfig:
    warn_unknown_keys(search_raw, KNOWN_SECTION_KEYS["search"], prefix="search", path=path)
    return SearchConfig(
        dense_top_k=parse_positive_int(
            search_raw.get("dense_top_k", 20), field="search.dense_top_k", path=path
        ),
        bm25_top_k=parse_positive_int(
            search_raw.get("bm25_top_k", 20), field="search.bm25_top_k", path=path
        ),
        final_top_k=parse_positive_int(
            search_raw.get("final_top_k", 5), field="search.final_top_k", path=path
        ),
        rrf_k=parse_positive_int(search_raw.get("rrf_k", 60), field="search.rrf_k", path=path),
        build_vector_index=parse_bool(
            search_raw.get("build_vector_index", True),
            field="search.build_vector_index",
            path=path,
        ),
    )


def load_config(path: str | Path | None = None) -> AppConfig:
    config_path = Path(path or os.environ.get("SNTX_SEM_CONFIG", "config.yaml"))
    if not config_path.is_file():
        cfg = AppConfig()
        cfg.resolve_paths(config_path.parent if config_path.name != "config.yaml" else Path.cwd())
        return cfg

    try:
        with config_path.open(encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        raise ConfigError(f"некорректный YAML: {exc}", path=config_path) from exc

    raw = validate_root(loaded, path=config_path)

    local_configs = [
        LocalConfig(path=item["path"], label=item["label"])
        for item in validate_local_configs(raw.get("local_configs"), path=config_path)
    ]

    llm_raw = require_mapping(raw.get("llm"), field="llm", path=config_path)
    embedding_raw = require_mapping(raw.get("embedding"), field="embedding", path=config_path)
    search_raw = require_mapping(raw.get("search"), field="search", path=config_path)
    java_raw = require_mapping(raw.get("java_exporter"), field="java_exporter", path=config_path)
    bsp_raw = require_mapping(raw.get("bsp"), field="bsp", path=config_path)
    mcp_raw = require_mapping(raw.get("mcp"), field="mcp", path=config_path)
    api_raw = require_mapping(raw.get("api"), field="api", path=config_path)

    warn_unknown_keys(llm_raw, KNOWN_SECTION_KEYS["llm"], prefix="llm", path=config_path)
    warn_unknown_keys(
        embedding_raw, KNOWN_SECTION_KEYS["embedding"], prefix="embedding", path=config_path
    )
    warn_unknown_keys(
        java_raw, KNOWN_SECTION_KEYS["java_exporter"], prefix="java_exporter", path=config_path
    )
    warn_unknown_keys(bsp_raw, KNOWN_SECTION_KEYS["bsp"], prefix="bsp", path=config_path)
    warn_unknown_keys(mcp_raw, KNOWN_SECTION_KEYS["mcp"], prefix="mcp", path=config_path)
    warn_unknown_keys(api_raw, KNOWN_SECTION_KEYS["api"], prefix="api", path=config_path)

    cfg = AppConfig(
        platform_version=parse_string(
            raw.get("platform_version", "8.3.27"),
            field="platform_version",
            path=config_path,
            allow_empty=False,
        ),
        hbk_dir=parse_path_value(raw.get("hbk_dir", "./hbk"), field="hbk_dir", path=config_path),
        data_dir=parse_path_value(
            raw.get("data_dir", "./data"), field="data_dir", path=config_path
        ),
        export_dir=parse_path_value(
            raw.get("export_dir", "./data/export"), field="export_dir", path=config_path
        ),
        index_dir=parse_path_value(
            raw.get("index_dir", "./data/index"), field="index_dir", path=config_path
        ),
        benchmark_results_path=parse_path_value(
            raw.get("benchmark_results_path", "./config/benchmark_results.yaml"),
            field="benchmark_results_path",
            path=config_path,
        ),
        local_configs=local_configs,
        llm=LLMConfig(
            provider=parse_string(
                llm_raw.get("provider", "openai_compatible"), field="llm.provider", path=config_path
            ),
            base_url=parse_string(
                llm_raw.get("base_url", ""), field="llm.base_url", path=config_path
            ),
            api_key=parse_string(llm_raw.get("api_key", ""), field="llm.api_key", path=config_path),
            api_key_env=parse_string(
                llm_raw.get("api_key_env", "OPENAI_API_KEY"),
                field="llm.api_key_env",
                path=config_path,
            ),
            model=parse_string(llm_raw.get("model", "auto"), field="llm.model", path=config_path),
        ),
        embedding=EmbeddingConfig(
            provider=validate_embedding_provider(
                embedding_raw.get("provider", ""), path=config_path
            ),
            model=parse_string(
                embedding_raw.get("model", "intfloat/multilingual-e5-base"),
                field="embedding.model",
                path=config_path,
                allow_empty=False,
            ),
            device=parse_string(
                embedding_raw.get("device", "cpu"), field="embedding.device", path=config_path
            ),
            base_url=parse_string(
                embedding_raw.get("base_url", ""), field="embedding.base_url", path=config_path
            ),
            api_key=parse_string(
                embedding_raw.get("api_key", ""), field="embedding.api_key", path=config_path
            ),
            api_key_env=parse_string(
                embedding_raw.get("api_key_env", "OPENAI_API_KEY"),
                field="embedding.api_key_env",
                path=config_path,
            ),
            batch_size=parse_positive_int(
                embedding_raw.get("batch_size", 64), field="embedding.batch_size", path=config_path
            ),
            timeout=parse_positive_float(
                embedding_raw.get("timeout", 120.0), field="embedding.timeout", path=config_path
            ),
            query_prefix=parse_string(
                embedding_raw.get("query_prefix", "query: "),
                field="embedding.query_prefix",
                path=config_path,
            ),
            passage_prefix=parse_string(
                embedding_raw.get("passage_prefix", "passage: "),
                field="embedding.passage_prefix",
                path=config_path,
            ),
        ),
        search=_load_search_config(search_raw, path=config_path),
        java_exporter=JavaExporterConfig(
            jar_path=parse_string(
                java_raw.get("jar_path", ""), field="java_exporter.jar_path", path=config_path
            ),
            enabled=parse_bool(
                java_raw.get("enabled", True), field="java_exporter.enabled", path=config_path
            ),
        ),
        bsp=BspConfig(
            path=parse_string(bsp_raw.get("path", ""), field="bsp.path", path=config_path),
            enabled=parse_bool(
                bsp_raw.get("enabled", False), field="bsp.enabled", path=config_path
            ),
        ),
        mcp=McpConfig(
            log_level=parse_string(
                mcp_raw.get("log_level", "INFO"), field="mcp.log_level", path=config_path
            ),
            log_file=parse_string(
                mcp_raw.get("log_file", ""), field="mcp.log_file", path=config_path
            ),
            log_max_chars=parse_positive_int(
                mcp_raw.get("log_max_chars", 2000), field="mcp.log_max_chars", path=config_path
            ),
        ),
        api=ApiConfig(
            host=parse_string(api_raw.get("host", "127.0.0.1"), field="api.host", path=config_path),
            port=parse_port(api_raw.get("port", 8000), path=config_path),
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
