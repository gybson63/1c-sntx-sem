---
name: 1c-syntax-sem-mcp
description: Use the 1c-syntax-sem MCP server to answer questions about 1C:Enterprise platform help, BSL syntax, SDBL query language, platform API, BSP API, local configuration examples, or to diagnose the MCP/search index behavior.
---

# 1c-syntax-sem MCP

## When To Use

Use this skill when the user asks about:

- 1C:Enterprise platform syntax, methods, objects, or platform API.
- BSL language syntax.
- SDBL query language.
- BSP API, especially exported methods from `#Область ПрограммныйИнтерфейс`.
- Examples from configured `local_configs`.
- MCP server behavior, search quality, index status, or Docker/local startup.

Do not treat arbitrary application configuration code as platform or BSP reference. Use `find_examples` only for examples from local configurations.

## Tool Workflow

Prefer MCP first. If MCP is unavailable, use HTTP API, then CLI as fallback.

1. Start with `search_help` for general questions.
2. Use `search_bsl_syntax` for BSL language syntax.
3. Use `search_query_language` for SDBL query language.
4. Open the best result with `get_topic`.
5. Use `find_examples` when implementation examples are requested.
6. Use `list_domains` to inspect index coverage or diagnose missing domains.

## MCP Tools

| Tool | Use |
| --- | --- |
| `search_help` | General semantic search: `query`, `domain`, `limit`. |
| `search_bsl_syntax` | BSL syntax help. |
| `search_query_language` | SDBL query language help. |
| `get_topic` | Full help topic by `topic_id`, optionally with examples. |
| `find_examples` | Code examples by `query` or `topic_id` from `local_configs`. |
| `list_domains` | Index statistics by domain. |

Supported domains include `all`, `bsl`, `query`, `bsp`, `bsl_lang`, `query_lang`, and `platform_api`.

## Run Modes

Use in-process MCP for local development:

```bash
python -m sntx_sem.mcp_server
```

Use thin HTTP MCP when the API runs in Docker or another process:

```bash
set SNTX_SEM_API_URL=http://localhost:8051
sntx-sem mcp
```

Typical Cursor MCP entries:

- Local: `python -m sntx_sem.mcp_server` with `SNTX_SEM_CONFIG` pointing to `config.yaml`.
- Docker/thin: `sntx-sem mcp` with `SNTX_SEM_API_URL=http://localhost:8051`.

## Diagnostics

Check readiness before blaming tool behavior:

```bash
python -m sntx_sem status
```

For HTTP mode, inspect:

- Web UI: `/admin`
- Logs endpoint: `GET /logs`
- Search endpoint: `POST /search`
- Topic endpoint: `GET /topic/{topic_id}`

If results are missing, check the domain first with `list_domains`, then verify that the relevant ingest step ran:

- Platform help: `python -m sntx_sem ingest --platform-path "..."`
- BSP API: `python -m sntx_sem ingest-bsp`
- Local examples: `python -m sntx_sem scan-examples`

## Answer Format

When answering from MCP results, include the practical answer first, then cite the source with `topic_id`, domain, and tool used. Mention uncertainty when the MCP result is sparse or when only local examples support the answer.
