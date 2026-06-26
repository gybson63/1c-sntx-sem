# Docker: sntx-sem

## Архитектура

Один образ, два режима запуска:

| Режим | Команда | Назначение |
|-------|---------|------------|
| API + Web-UI | `docker compose up -d` | HTTP на порту 8051 (по умолчанию), браузер |
| MCP | `docker compose run --rm -i sntx-sem python -m sntx_sem.mcp_server` | Cursor stdio |

Общие volumes: `config.yaml`, `./data`, `./hbk`, том `model-cache` (кэш E5).

## Первый запуск

```powershell
cp config.docker.example.yaml config.yaml
# Скопируйте HBK из платформы 1С в ./hbk
docker compose build
docker compose run --rm sntx-sem sntx-sem ingest --hbk-dir /hbk
# или ingest на хосте, затем:
docker compose up -d
```

Web-UI: http://localhost:8051  
Health: http://localhost:8051/health

MCP: скопируйте [mcp.json.docker.example](../mcp.json.docker.example) и укажите путь к `docker-compose.yml`.

## Переменные окружения (.env)

```env
SNTX_SEM_PORT=8051
SNTX_SEM_DATA=./data
SNTX_SEM_HBK=./hbk
```

На Windows используйте прямые слэши в путях: `C:/Git/1c-sntx-sem/data`.

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
| `config.yaml` embedding | Нет | `index --rebuild` |
| Только HBK / export | Нет | `ingest` / `index` |

\* Если менялась только логика поиска без смены эмбеддингов.

## Что сохраняется

| Путь / том | Содержимое |
|------------|------------|
| `./data/` | export, LanceDB index |
| `./hbk/` | HBK платформы |
| `./config.yaml` | настройки |
| том `model-cache` | веса E5 (Hugging Face cache) |

`docker compose down` не удаляет именованный том `model-cache`.

## Полная переиндексация

```powershell
docker compose run --rm sntx-sem sntx-sem index --rebuild
```

## Ограничения

- Одновременно `compose up` (API) и `compose run -i` (MCP) загружают E5 в память дважды.
- Следующая итерация: тонкий MCP на хосте через `SNTX_SEM_API_URL` (один контейнер с API).
