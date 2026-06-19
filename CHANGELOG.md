# Changelog

Все значимые изменения проекта документируются в этом файле.

Формат основан на [Keep a Changelog](https://keepachangelog.com/ru/1.1.0/),
версионирование — [Semantic Versioning](https://semver.org/lang/ru/).

Опубликованные версии — в [GitHub Releases](https://github.com/gybson63/1c-sntx-sem/releases).

## [Unreleased]

### Added

- Логирование MCP-запросов и ответов инструментов (stderr, опционально в файл через `SNTX_SEM_MCP_LOG_FILE`).
- Инфраструктура разработки: ruff, mypy, pre-commit, CHANGELOG, release/CI workflows.
- `scripts/release.py`, `scripts/check_changelog.py`, `src/sntx_sem/_version.py`.
- Правила workflow в `.cursor/rules/development-workflow.mdc`.

### Changed

### Fixed

- Исправления типизации и lint для прохождения ruff/mypy.

### Removed

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
