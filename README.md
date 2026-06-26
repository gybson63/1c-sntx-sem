# MCP-сервер семантической справки 1С

Семантический поиск по справке платформы 1С (BSL + язык запросов SDBL) с примерами из локальных конфигураций. MCP-сервер для Cursor и Claude Desktop.

**База справки не включена в репозиторий** — каждый пользователь собирает её локально из своей лицензированной платформы 1С.

## Быстрый старт

```bash
git clone https://github.com/gybson63/1c-sntx-sem.git
cd 1c-sntx-sem
pip install -e ".[embeddings]"   # локальная E5; для OpenAI API достаточно pip install -e .
cp config.yaml.example config.yaml
# отредактируйте config.yaml: local_configs и при необходимости пути

# 1. Извлечь справку и построить векторный индекс (~5–15 мин на CPU)
python -m sntx_sem ingest --platform-path "C:/Program Files/1cv8/8.3.27.xxxx/bin"

# 2. (опционально) API БСП из XML-выгрузки (индекс пересобирается автоматически)
python -m sntx_sem ingest-bsp --bsp-dir ./examples/БСП

# 3. (опционально) Примеры из конфигураций
python -m sntx_sem scan-examples

# 4. Проверка
python -m sntx_sem status   # ready: true
python -m sntx_sem search "левое соединение" --domain query
python -m sntx_sem search "вариант отчета" --domain bsp
```

По умолчанию эмбеддинги считаются локально (E5 через `sentence_transformers`, ~120 MB при первом запуске).  
Команды `ingest` и `ingest-bsp` сразу строят индекс; флаг `--no-index` — только экспорт JSONL.  
Ручной пересбор: `python -m sntx_sem index --rebuild`.

## HTTP API и Web-UI

```bash
sntx-sem serve
# http://127.0.0.1:8000 — поиск по справке в браузере
```

## Docker

Один образ, два режима: долгоживущий **API + Web-UI** и отдельный запуск **MCP** для Cursor.

```bash
cp config.docker.example.yaml config.yaml
# HBK в ./hbk, ingest/index на хосте или в контейнере
docker compose build
docker compose up -d
# Web-UI: http://localhost:8051
```

MCP в Docker: [mcp.json.docker.example](mcp.json.docker.example).  
Подробнее: [docs/DOCKER.md](docs/DOCKER.md).

## MCP в Cursor

Пример конфигурации: [examples/cursor-mcp.json](examples/cursor-mcp.json) или [`.cursor/mcp.json`](.cursor/mcp.json) в корне репозитория. В MCP указывается только путь к **`config.yaml`** — все настройки проекта там.

Правило для агента: [`.cursor/rules/1c-syntax-sem.mdc`](.cursor/rules/1c-syntax-sem.mdc) — когда вызывать `search_help`, `get_topic` и др.

Логи MCP: секция `mcp:` в `config.yaml` или env `SNTX_SEM_MCP_LOG_FILE`.

```json
{
  "mcpServers": {
    "1c-syntax-sem": {
      "command": "python",
      "args": ["-m", "sntx_sem.mcp_server"],
      "env": {
        "SNTX_SEM_CONFIG": "C:/path/to/1c-sntx-sem/config.yaml"
      }
    }
  }
}
```

Пример: [examples/cursor-mcp.json](examples/cursor-mcp.json)

## Возможности

- Гибридный поиск: embeddings + BM25 + RRF
- Провайдеры эмбеддингов: `sentence_transformers` / E5 (по умолчанию), OpenAI-compatible API, Ollama
- HTTP API (`serve`) и минимальный Web-UI
- Docker: локальная E5 в контейнере, том кэша модели
- Домены: `platform_api`, `bsl_lang`, `query_lang`, `bsp` (Библиотека стандартных подсистем)
- MCP: `search_help`, `get_topic`, `find_examples`, …
- Примеры из локальных конфигураций (опционально)
- Java exporter для расширенного экспорта BSL (опционально)
- Бенчмарк LLM для выбора модели

## API БСП (Библиотека стандартных подсистем)

Выгрузите конфигурацию БСП в XML и укажите путь в `config.yaml`:

```yaml
bsp:
  path: "./examples/БСП"
  enabled: true
```

```bash
python -m sntx_sem ingest-bsp
python -m sntx_sem search "СформироватьСообщение" --domain bsp
```

Индексируются экспортные методы из `#Область ПрограммныйИнтерфейс` в `CommonModules`.  
ID топика: `bsp:ИмяМодуля.ИмяМетода` (например `bsp:ВариантыОтчетов.ВариантОтчета`).

## Конфигурация

**Все настройки проекта — в `config.yaml`** (не коммитится в git).

```bash
python -m sntx_sem status   # config: настройки, index: состояние базы
```

## Эмбеддинги

По умолчанию — локальная модель E5 (`intfloat/multilingual-e5-small`, провайдер `sentence_transformers`).  
Установка: `pip install -e ".[embeddings]"`. Без этого extra работает только OpenAI-compatible API.

```yaml
embedding:
  provider: sentence_transformers
  model: intfloat/multilingual-e5-small
  device: cpu
```

```yaml
# OpenAI-compatible API (Polza.ai, OpenAI, …) — pip install -e .
embedding:
  provider: openai_compatible
  base_url: "https://polza.ai/api/v1"
  api_key: "pza_..."              # ключ прямо в config.yaml
  model: "text-embedding-3-small"
  query_prefix: ""
  passage_prefix: ""

# Ollama (localhost)
embedding:
  provider: ollama
  model: "nomic-embed-text"
```

Альтернатива `api_key`: `api_key_env: "POLZA_AI_API_KEY"` и ключ в env.

После смены провайдера или модели выполните `python -m sntx_sem index --rebuild`.  
`python -m sntx_sem status` предупредит, если config не совпадает с собранным индексом.

## Локальные конфигурации (примеры)

В `config.yaml`:

```yaml
local_configs:
  - path: "C:/Projects/MyConfig/edt-src"
    label: "MyConfig"
```

```bash
python -m sntx_sem scan-examples
python -m sntx_sem scan-examples --link   # LLM linking (нужен API key)
```

## Java exporter (опционально)

```bash
cd tools/bsl-context-exporter
./gradlew jar
```

В `config.yaml` установите `java_exporter.enabled: true`.

## Структура

```
data/                     # локальная база (не в git)
  export/all_chunks.jsonl
  index/                  # LanceDB + build_meta.json
src/sntx_sem/             # Python + MCP
tools/bsl-context-exporter/
hbk/                      # HBK из платформы (не в git)
benchmarks/               # LLM ranker
```

Подробнее: [data/README.md](data/README.md)

## MCP Tools

| Tool | Описание |
|------|----------|
| `search_help` | Семантический поиск |
| `search_bsl_syntax` | BSL (`shlang`) |
| `search_query_language` | Язык запросов |
| `get_topic` | Страница + примеры |
| `find_examples` | Примеры из конфигов |
| `list_domains` | Статистика индекса |

## Авторские права

Этот проект — **инструмент** для работы с вашей лицензированной платформой 1С:Предприятие.

- HBK-файлы справки и извлечённый текст принадлежат правообладателю (1С).
- Используйте только файлы из **вашей** установки платформы.
- **Не распространяйте** содержимое каталога `data/` (экспорт, индекс) третьим лицам.

Исходный код инструментов распространяется под лицензией MIT — см. [LICENSE](LICENSE).

## Тесты

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

Тесты HBK требуют локальный файл `hbk/shquery_root.hbk` (из установки платформы).

## Разработка

```bash
pip install -e ".[dev]"
pre-commit install
pre-commit install --hook-type commit-msg
```

Перед коммитом: `ruff check .`, `ruff format .`, `mypy src`, `pytest tests/ -v`.

Workflow: feature-ветка → PR → релиз через `python scripts/release.py prepare X.Y.Z`.
Подробнее: [`.cursor/rules/development-workflow.mdc`](.cursor/rules/development-workflow.mdc).
