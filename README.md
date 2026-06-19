# MCP-сервер семантической справки 1С

Семантический поиск по справке платформы 1С (BSL + язык запросов SDBL) с примерами из локальных конфигураций. MCP-сервер для Cursor и Claude Desktop.

**База справки не включена в репозиторий** — каждый пользователь собирает её локально из своей лицензированной платформы 1С.

## Быстрый старт

```bash
git clone https://github.com/gybson63/1c-sntx-sem.git
cd 1c-sntx-sem
pip install -e .
cp config.yaml.example config.yaml
# отредактируйте config.yaml: local_configs и при необходимости пути

# 1. Извлечь справку из установленной платформы 1С
python -m sntx_sem ingest --platform-path "C:/Program Files/1cv8/8.3.27.xxxx/bin"

# 2. Построить индекс (~5–15 мин на CPU, скачается модель embeddings ~120 MB)
python -m sntx_sem index --rebuild

# 3. (опционально) Примеры из конфигураций
python -m sntx_sem scan-examples

# 4. Проверка
python -m sntx_sem status   # ready: true
python -m sntx_sem search "левое соединение" --domain query
```

## MCP в Cursor

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

- Гибридный поиск: E5 embeddings + BM25 + RRF
- Домены: `platform_api`, `bsl_lang`, `query_lang`
- MCP: `search_help`, `get_topic`, `find_examples`, …
- Примеры из локальных конфигураций (опционально)
- Java exporter для расширенного экспорта BSL (опционально)
- Бенчмарк LLM для выбора модели

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
  index/                  # LanceDB
  manifest.yaml
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
