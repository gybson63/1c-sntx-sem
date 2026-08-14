# ADR 0003: Optional IVF vector index in LanceDB

- Status: Accepted
- Date: 2026-07-07

## Context

Для крупных наборов данных IVF-индекс ускоряет dense search, но требует больше RAM при rebuild и в рантайме.
В Docker-окружении пользователи часто ограничены по памяти, что приводит к OOM при `create_index`.

Нужен управляемый компромисс между скоростью поиска и доступностью на слабых машинах.

## Decision

Сделать построение IVF-индекса опциональным через `search.build_vector_index`:

- `true`: строить IVF (при достаточной RAM);
- `false`: использовать Lance таблицу без IVF.

Сохранять признак `vector_index_built` в `build_meta.json` и показывать состояние в `/status`/`/health`.

## Consequences

### Positive

- Проект стабильно работает в low-memory Docker сценариях.
- Пользователь явно управляет trade-off `скорость <-> ресурсы`.
- Диагностика сообщает о missing IVF и дает actionable рекомендации.

### Negative

- Без IVF latency на dense search выше, особенно на `domain=all`.
- Возрастает вариативность производительности между окружениями.

### Follow-up

- Добавить бенчмарк-профили с фиксированными профилями RAM/latency для рекомендаций по умолчанию.
