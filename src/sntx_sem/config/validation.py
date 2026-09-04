"""Strict validation helpers for YAML configuration."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from sntx_sem.config.embedding import EMBEDDING_PROVIDER_CHOICES, normalize_embedding_provider

logger = logging.getLogger(__name__)

KNOWN_ROOT_KEYS = frozenset(
    {
        "platform_version",
        "hbk_dir",
        "data_dir",
        "export_dir",
        "index_dir",
        "benchmark_results_path",
        "local_configs",
        "llm",
        "embedding",
        "search",
        "java_exporter",
        "bsp",
        "mcp",
        "api",
    }
)

KNOWN_SECTION_KEYS = {
    "llm": frozenset({"provider", "base_url", "api_key", "api_key_env", "model"}),
    "embedding": frozenset(
        {
            "provider",
            "model",
            "device",
            "base_url",
            "api_key",
            "api_key_env",
            "batch_size",
            "timeout",
            "query_prefix",
            "passage_prefix",
        }
    ),
    "search": frozenset(
        {"dense_top_k", "bm25_top_k", "final_top_k", "rrf_k", "build_vector_index"}
    ),
    "java_exporter": frozenset({"jar_path", "enabled"}),
    "bsp": frozenset({"path", "enabled"}),
    "mcp": frozenset({"log_level", "log_file", "log_max_chars"}),
    "api": frozenset({"host", "port"}),
}


class ConfigError(ValueError):
    """Configuration validation error with optional field path."""

    def __init__(self, message: str, *, field: str | None = None, path: Path | None = None) -> None:
        self.field = field
        self.path = path
        parts: list[str] = []
        if path is not None:
            parts.append(str(path))
        if field:
            parts.append(field)
        prefix = ": ".join(parts)
        super().__init__(f"{prefix}: {message}" if prefix else message)


def require_mapping(value: Any, *, field: str, path: Path | None = None) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConfigError("ожидается объект (mapping)", field=field, path=path)
    return value


def parse_bool(value: Any, *, field: str, path: Path | None = None) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "1", "on"}:
            return True
        if normalized in {"false", "no", "0", "off"}:
            return False
    raise ConfigError(f"ожидается boolean, получено {value!r}", field=field, path=path)


def parse_positive_int(value: Any, *, field: str, path: Path | None = None) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(
            f"ожидается целое число, получено {value!r}", field=field, path=path
        ) from exc
    if number <= 0:
        raise ConfigError("значение должно быть > 0", field=field, path=path)
    return number


def parse_positive_float(value: Any, *, field: str, path: Path | None = None) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"ожидается число, получено {value!r}", field=field, path=path) from exc
    if number <= 0:
        raise ConfigError("значение должно быть > 0", field=field, path=path)
    return number


def parse_port(value: Any, *, field: str = "api.port", path: Path | None = None) -> int:
    try:
        port = int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(
            f"ожидается порт 1..65535, получено {value!r}", field=field, path=path
        ) from exc
    if not 1 <= port <= 65535:
        raise ConfigError(f"порт вне диапазона 1..65535: {port}", field=field, path=path)
    return port


def parse_path_value(value: Any, *, field: str, path: Path | None = None) -> Path:
    if isinstance(value, Path):
        return value
    if isinstance(value, str) and value.strip():
        return Path(value)
    raise ConfigError(f"ожидается путь (строка), получено {value!r}", field=field, path=path)


def parse_string(
    value: Any, *, field: str, path: Path | None = None, allow_empty: bool = True
) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ConfigError(
            f"ожидается строка, получено {type(value).__name__}", field=field, path=path
        )
    if not allow_empty and not value.strip():
        raise ConfigError("строка не должна быть пустой", field=field, path=path)
    return value


def warn_unknown_keys(
    raw: dict[str, Any],
    known: frozenset[str],
    *,
    prefix: str,
    path: Path | None = None,
) -> None:
    for key in raw:
        if key not in known:
            location = f"{path}: " if path else ""
            logger.warning("%sнеизвестный ключ конфигурации: %s.%s", location, prefix, key)


def validate_root(raw: Any, *, path: Path | None = None) -> dict[str, Any]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ConfigError("корень YAML должен быть объектом", path=path)
    warn_unknown_keys(raw, KNOWN_ROOT_KEYS, prefix="root", path=path)
    return raw


def validate_local_configs(value: Any, *, path: Path | None = None) -> list[dict[str, str]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ConfigError("ожидается список", field="local_configs", path=path)
    result: list[dict[str, str]] = []
    for index, item in enumerate(value):
        field = f"local_configs[{index}]"
        mapping = require_mapping(item, field=field, path=path)
        if "path" not in mapping or "label" not in mapping:
            raise ConfigError("нужны поля path и label", field=field, path=path)
        result.append(
            {
                "path": parse_string(
                    mapping["path"], field=f"{field}.path", path=path, allow_empty=False
                ),
                "label": parse_string(
                    mapping["label"], field=f"{field}.label", path=path, allow_empty=False
                ),
            }
        )
    return result


def validate_embedding_provider(value: Any, *, path: Path | None = None) -> str:
    provider = parse_string(value, field="embedding.provider", path=path)
    if not provider:
        return ""
    normalized = normalize_embedding_provider(provider)
    if normalized not in EMBEDDING_PROVIDER_CHOICES and provider != "huggingface":
        raise ConfigError(
            f"неизвестный provider: {provider!r}",
            field="embedding.provider",
            path=path,
        )
    return provider
