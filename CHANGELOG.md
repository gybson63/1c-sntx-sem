# Changelog

Все значимые изменения проекта документируются в этом файле.

Формат основан на [Keep a Changelog](https://keepachangelog.com/ru/1.1.0/),
версионирование — [Semantic Versioning](https://semver.org/lang/ru/).

Опубликованные версии — в [GitHub Releases](https://github.com/gybson63/1c-sntx-sem/releases).

## [Unreleased]

## [0.2.0] - 2026-06-26

### Added

- Тонкий MCP (`sntx-sem mcp`): прокси к HTTP API через `SNTX_SEM_API_URL` — один контейнер с E5, MCP на хосте без torch.
- HTTP API: `/status`, `POST /examples`, `/settings/embedding`, `/settings/embedding/test`, `/logs`.
- Фоновые задачи: `POST /jobs/ingest`, `/jobs/ingest-bsp`, `/jobs/index`, `GET /jobs/{id}`.
- Web-UI админка (`/admin`): статус базы, ingest/index, test embedding.
- Документация API: [docs/API.md](docs/API.md).
- CI: docker-smoke job (build + healthcheck).
- Optional extra `[mcp]` — документированный режим thin MCP без `[embeddings]`.
- HTTP API (`sntx-sem serve`): `/health`, `/search`, `/topic/{id}`, `/stats`; минимальный Web-UI.
- Docker: образ с локальной E5, `docker-compose.yml`, том `model-cache`, примеры `config.docker.example.yaml` и `mcp.json.docker.example`.
- Настраиваемые провайдеры эмбеддингов: `sentence_transformers` / E5 (по умолчанию), OpenAI-compatible API, Ollama (`embedding.provider`, `base_url`, `api_key` в config).
- Индексация публичного API БСП (`ingest-bsp`): парсинг комментариев экспортных методов из `#Область ПрограммныйИнтерфейс`, домен `bsp` в `search_help` / `get_topic`.
- Логирование MCP-запросов и ответов инструментов (stderr, опционально в файл через `SNTX_SEM_MCP_LOG_FILE`).
- Инфраструктура разработки: ruff, mypy, pre-commit, CHANGELOG, release/CI workflows.
- `scripts/release.py`, `scripts/check_changelog.py`, `src/sntx_sem/_version.py`.
- Правила workflow в `.cursor/rules/development-workflow.mdc`.

### Changed

- `sentence-transformers` вынесен в optional extra `[embeddings]`; базовый `pip install` — для OpenAI API без torch.
- Провайдер локальных эмбеддингов: `sentence_transformers` (алиас `huggingface`).
- OpenAI-compatible embedding backend: переиспользование HTTP-сессии и LRU-кэш `embed_query` (~10× быстрее повторных запросов к API).
- MCP-сервер: фоновая предзагрузка индекса (LanceDB + BM25) при старте — первый `search_help` без паузы ~4 с.
- Поиск с фильтром домена: pre-filter в LanceDB (`.where(domain=…)`) и BM25 только по чанкам выбранного домена; IVF + scalar index на `domain`/`vector` при сборке и при первой загрузке старого индекса.
- Метаданные сборки индекса — в `data/index/build_meta.json` (рядом с LanceDB), не в отдельном `manifest.yaml`.
- `status` показывает `config` (из config.yaml) и `index` (из build_meta) отдельно; корректно определяет `embedding_mismatch`.
- `ingest` и `ingest-bsp` автоматически строят векторный индекс; флаг `--no-index` для экспорта без эмбеддингов.
- В `chunks_meta.json` сохраняются поля `parameters` и `signature` для полного ответа `get_topic`.

### Fixed

- Создание каталога для `SNTX_SEM_MCP_LOG_FILE`, если он ещё не существует.

### Removed

- `data/manifest.yaml` — дублировал состояние индекса и путал с настройками в `config.yaml`.

## [0.1.0] - 2026-06-19

### Added

- MCP-сервер семантического поиска по справке платформы 1С (BSL, язык запросов, platform API).
- CLI: `ingest`, `index`, `status`, `search`, `scan-examples`.
- Гибридный поиск: E5 embeddings + BM25 + RRF (LanceDB).
- Сканер примеров кода из локальных конфигураций с опциональным LLM linking.
- Java exporter для расширенного экспорта BSL (`tools/bsl-context-exporter`).
- Бенчмарк LLM ranker.

### Changed

- База справки не включается в репозиторий — пользователи собирают её локально из HBK своей платформы 1С.
