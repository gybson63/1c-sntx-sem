# ADR 0002: Hybrid search with semantic and lexical text split

- Status: Accepted
- Date: 2026-07-07

## Context

Один канал поиска дает нестабильное качество:

- только BM25 плохо покрывает разговорные запросы;
- только dense хуже отрабатывает точные совпадения имен методов/типов.

Для 1С-справки важно одновременно учитывать:

- семантику описания и параметров;
- точные идентификаторы, названия, путь/модуль.

## Decision

Использовать гибридный поиск:

- `semantic_text` для embeddings;
- `lexical_text` для BM25 и title-aware сигналов;
- объединение dense и BM25 через RRF;
- дополнительные бонусы (`title_bonus`, `semantic_intent_bonus`) для финального ранжирования.

Итоговые результаты включают explainability поля (`match_sources`, `score_breakdown`, `match_explanation`).

## Consequences

### Positive

- Выше релевантность как для естественных запросов, так и для name-like запросов.
- Понятная декомпозиция score для диагностики качества ранжирования.
- Более устойчивый результат для `domain=all` и смешанных доменов.

### Negative

- Логика ранжирования распределена по `index/ranking.py` и `index/explain.py`; изменения формул/бонусов требуют аккуратной регрессии по search quality.

### Follow-up

- ~~Рассмотреть вынос fusion/bonus/explainability в отдельные модули.~~ Выполнено в P1 (`ranking.py`, `explain.py`, `tokens.py`, `chunk_text.py`).
