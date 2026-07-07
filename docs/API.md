# HTTP API

Базовый URL: `http://127.0.0.1:8000` (локально) или `http://localhost:8051` (Docker).

Интерактивная документация: `/docs` (OpenAPI/Swagger).

## Health и статус

| Method | Path | Описание |
|--------|------|----------|
| GET | `/health` | Краткий healthcheck (`ready`, `version`, embedding) |
| GET | `/status` | Полный статус базы (`bundled_database_status`) |
| GET | `/stats` | Статистика доменов индекса |
| GET | `/logs` | Хвост логов API (`?since=0&limit=200`) |

## Поиск

| Method | Path | Body |
|--------|------|------|
| POST | `/search` | `{ "query": "...", "domain": "all", "limit": 5 }` |
| GET | `/topic/{topic_id}` | `?include_examples=true` |
| POST | `/examples` | `{ "query"?: "...", "topic_id"?: "...", "limit": 5 }` |

Домены `domain`: `all`, `bsl`, `query`, `bsp`, `platform_api`, …

Ответ `POST /search` — массив результатов:

```json
{
  "id": "platform:strsplit",
  "domain": "platform_api",
  "title": "СтрРазделить",
  "score": 0.8421,
  "entity_kind": "method",
  "html_path": "...",
  "excerpt": "Фрагмент полного текста...",
  "excerpt_start": 120,
  "excerpt_end": 620,
  "highlight_terms": ["строку", "массив"],
  "match_sources": ["semantic", "bm25", "intent"],
  "match_explanation": "Семантический поиск: rank 2; BM25: rank 4; описание совпало с намерением запроса.",
  "semantic_excerpt": "Возвращает массив строк, полученных разделением исходной строки...",
  "semantic_highlight_terms": ["массив", "строк", "разделением"],
  "score_breakdown": {
    "total": 0.8421,
    "dense_rrf": 0.0214,
    "dense_similarity": 0.7312,
    "bm25_rrf": 0.0161,
    "title_bonus": 0.008,
    "semantic_intent_bonus": 0.06,
    "dense_rank": 2,
    "bm25_rank": 4,
    "dense_distance": 0.367,
    "bm25_raw": 1.42
  }
}
```

`match_sources` показывает, какие части hybrid search внесли вклад: `semantic` (dense vector search), `bm25` (лексический поиск), `title` (бонус названия), `intent` (бонус совпадения смысловых основ в `semantic_text`). `score_breakdown` предназначен для анализа качества ранжирования; значения score округляются в HTTP-ответе.

## Эмбеддинги

| Method | Path | Описание |
|--------|------|----------|
| GET | `/settings/embedding` | Текущие настройки и `embedding_mismatch` |
| PUT | `/settings/embedding` | Сохранить настройки в `config.yaml` (пустой `api_key` — не менять) |
| POST | `/settings/embedding/test` | `{ "text": "..." }` → `{ model, dimensions }` |

## Фоновые задачи

| Method | Path | Описание |
|--------|------|----------|
| POST | `/jobs/ingest` | Ingest HBK из `config.hbk_dir` + rebuild index |
| POST | `/jobs/ingest-bsp` | Ingest BSP + rebuild (нужен `bsp.path`) |
| POST | `/jobs/index` | `{ "rebuild": true }` |
| GET | `/jobs/{job_id}` | Статус и лог (`?since_log=0`) |

## Config: `search.build_vector_index`

В `config.yaml` секция `search`:

- `build_vector_index: true` — после rebuild строится IVF-индекс LanceDB (быстрее поиск, нужно **8 ГБ+ RAM** в Docker).
- `build_vector_index: false` — только таблица LanceDB без IVF (default в `config.docker.example.yaml`, rebuild на ~4 ГБ).

Поиск с `domain=all` без IVF нагружает RAM сильнее, чем поиск в одном домене: один полный Lance scan + корпус в памяти. Для стабильного поиска по всем доменам в Docker выделите **8 ГБ+ RAM** (Docker Desktop → Settings → Resources) или выберите конкретный домен в Web-UI.

`GET /status` → `index.vector_index_built` и issue `vector_index_missing`, если IVF ожидался, но не построен.

## Web-UI

| Path | Описание |
|------|----------|
| `/` | Поиск по справке |
| `/admin` | Статус, настройки эмбеддингов, ingest/index, test embedding |

## Thin MCP

MCP на хосте проксирует эти эндпоинты:

```bash
export SNTX_SEM_API_URL=http://localhost:8051
sntx-sem mcp
```

См. [`mcp.json.docker.example`](../mcp.json.docker.example).
