# Локальная база справки

Каталог `data/` **не хранится в git**. Каждый пользователь собирает базу из своей лицензированной платформы 1С.

## Сборка

```bash
python -m sntx_sem ingest --platform-path "C:/Program Files/1cv8/8.3.27.xxxx/bin"
python -m sntx_sem index --rebuild
python -m sntx_sem status   # ready: true
```

## Что появится локально

| Путь | Назначение |
|------|------------|
| `export/all_chunks.jsonl` | Текст страниц справки |
| `index/` | LanceDB + embeddings + BM25 metadata |
| `manifest.yaml` | Версия платформы, дата сборки |
| `examples.jsonl` | (опционально) примеры из конфигураций |

Шаблон манифеста: [`manifest.yaml.example`](manifest.yaml.example).

## Авторские права

HBK-файлы и извлечённый текст справки — продукт 1С. Не распространяйте содержимое `data/` третьим лицам.
