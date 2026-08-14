# ADR 0001: Dual MCP modes (in-process and thin HTTP)

- Status: Accepted
- Date: 2026-07-07

## Context

Проект используется в разных средах:

- локальная разработка, где доступен Python runtime и локальные embeddings;
- Docker-сценарий, где тяжелые зависимости (torch/E5) лучше держать в контейнере;
- IDE/MCP клиенты ожидают stdio MCP сервер.

Нужна единая функциональность MCP tools при разной модели деплоя.

## Decision

Поддерживать два режима MCP:

1. **In-process** (`python -m sntx_sem.mcp_server`)
   - прямой доступ к `HelpSearchService` и локальному индексу.
2. **Thin HTTP** (`sntx-sem mcp` + `SNTX_SEM_API_URL`)
   - MCP stdio сервер проксирует tools в HTTP API через `SntxSemApiClient`.

В CLI (`mcp` команда) выбор режима происходит автоматически по `SNTX_SEM_API_URL`.

## Consequences

### Positive

- Гибкий deployment: локально и в Docker без изменения клиентского workflow.
- Снижение требований к host-machine в thin-режиме.
- Единый набор MCP tools независимо от runtime.

### Negative

- Параллельные реализации tools и обработки ошибок в двух файлах.
- Риск drift между контрактами in-process и thin.

### Follow-up

- Поддерживать contract tests на паритет инструментов между режимами.
