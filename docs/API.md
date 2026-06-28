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
