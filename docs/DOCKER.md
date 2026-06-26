# Docker: sntx-sem

## Архитектура

Рекомендуемый режим: **один контейнер** с HTTP API + Web-UI, MCP на хосте как тонкий клиент.

| Компонент | Команда | Назначение |
|-----------|---------|------------|
| API + Web-UI | `docker compose up -d` | HTTP на порту 8051, браузер |
| MCP (тонкий) | `sntx-sem mcp` на хосте | Cursor stdio → HTTP API |

```bash
# 1. Контейнер с API
docker compose up -d

# 2. MCP в Cursor (mcp.json.docker.example)
# SNTX_SEM_API_URL=http://localhost:8051
```

Альтернатива: in-process MCP в Docker (`docker compose run -i sntx-sem python -m sntx_sem.mcp_server`) — загружает E5 повторно.

Общие volumes: `config.yaml`, `./data`, `./hbk`, том `model-cache` (кэш E5).

## Первый запуск

```powershell
cp config.docker.example.yaml config.yaml
# Скопируйте HBK из платформы 1С в ./hbk
docker compose build
docker compose run --rm sntx-sem sntx-sem ingest --hbk-dir /hbk
docker compose up -d
```

Web-UI: http://localhost:8051  
Админка: http://localhost:8051/admin  
Health: http://localhost:8051/health

MCP: [`mcp.json.docker.example`](../mcp.json.docker.example) — `sntx-sem mcp` + `SNTX_SEM_API_URL`.

## Переменные окружения (.env)

```env
SNTX_SEM_PORT=8051
SNTX_SEM_DATA=./data
SNTX_SEM_HBK=./hbk
```

## Обновление после git pull

```powershell
docker compose down
docker compose build
docker compose up -d
```

| Изменение | Пересборка образа | Переиндексация |
|-----------|-------------------|----------------|
| Код `src/` | Да | Нет* |
| `pyproject.toml` / Dockerfile | Да (`--no-cache` при смене deps) | Нет* |
| `config.yaml` embedding | Нет | `index --rebuild` или `/admin` |
| Только HBK / export | Нет | ingest / index |

\* Если менялась только логика поиска без смены эмбеддингов.

## API

См. [docs/API.md](API.md).

## Что сохраняется

| Путь / том | Содержимое |
|------------|------------|
| `./data/` | export, LanceDB index |
| `./hbk/` | HBK платформы |
| `./config.yaml` | настройки |
| том `model-cache` | веса E5 (Hugging Face cache) |

## Полная переиндексация

```powershell
docker compose run --rm sntx-sem sntx-sem index --rebuild
```

Или через Web-UI: http://localhost:8051/admin
