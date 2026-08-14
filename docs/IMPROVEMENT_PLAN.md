# План улучшений архитектуры

Источник: раздел `Приоритетный план улучшений` из [`architect.md`](../architect.md).

## Цели

- Снизить архитектурную связность ядра поиска.
- Повысить модульность и тестируемость ключевых подсистем.
- Закрыть пробелы в инженерных quality-gates (coverage/security).

## P1 (краткосрочно)

- [x] Декомпозировать `HelpIndex` (`src/sntx_sem/index/store.py`) на отдельные модули:
  - [x] индексатор/хранилище (`chunk_text.py`, `storage.py`);
  - [x] ранжирование/fusion (`ranking.py`);
  - [x] explainability (`explain.py`).
- [x] Унифицировать CLI-путь поиска через `HelpSearchService` вместо прямого вызова `HelpIndex`.
- [x] Добавить contract tests на паритет MCP режимов:
  - [x] in-process (`src/sntx_sem/mcp_server.py`);
  - [x] thin (`src/sntx_sem/mcp/stdio_server.py`).

## P2 (среднесрочно)

- [ ] Разделить `config.py` на подсистемы:
  - [ ] loader/validation;
  - [ ] settings persistence;
  - [ ] database diagnostics.
- [ ] Добавить coverage-отчет и порог покрытия в CI.
- [ ] Вернуть интеграционный индексный тест в автоматический контур:
  - [ ] в основной CI, либо
  - [ ] в nightly workflow.

## P3 (долгосрочно)

- [ ] Поддерживать ADR-процесс для архитектурных решений (`docs/adr/`).
- [ ] Добавить security job в CI (dependency + static security scanning).
- [ ] Оценить и при необходимости внедрить персистентный backend для jobs.

## Контроль выполнения

- Архитектурный надзор обязателен через sub-agent `architect`:
  - [`.cursor/agents/architect.md`](../.cursor/agents/architect.md)
  - [`.cursor/rules/architecture-governance.mdc`](../.cursor/rules/architecture-governance.mdc)
