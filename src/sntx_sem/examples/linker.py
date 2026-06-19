"""LLM-based linking of code examples to help topics."""

from __future__ import annotations

import json
from pathlib import Path

from sntx_sem.examples.scanner import CodeExample
from sntx_sem.llm.client import LLMClient, resolve_model_for_task


LINK_PROMPT = """You are a 1C:Enterprise expert. Match the code fragment to help topic IDs.

Topics (JSON array of {{id, title, domain}}):
{topics}

Code fragment:
```
{code}
```

Query text (if any):
{query}

Return JSON only:
{{"topic_ids": ["id1"], "relevance": 0.0-1.0, "summary": "brief explanation in Russian"}}
"""


def link_example_with_llm(
    example: CodeExample,
    topic_catalog: list[dict],
    client: LLMClient,
    max_topics: int = 30,
) -> CodeExample:
    catalog = topic_catalog[:max_topics]
    messages = [
        {
            "role": "user",
            "content": LINK_PROMPT.format(
                topics=json.dumps(
                    [
                        {
                            "id": t["id"],
                            "title": t.get("title_ru") or t.get("title_en"),
                            "domain": t.get("domain"),
                        }
                        for t in catalog
                    ],
                    ensure_ascii=False,
                ),
                code=example.code[:1500],
                query=example.query_text or "(none)",
            ),
        }
    ]
    try:
        result = client.chat_json(messages)
        example.topic_ids = result.get("topic_ids", [])
        example.relevance = float(result.get("relevance", 0))
        example.summary = result.get("summary", "")
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        example.topic_ids = []
        example.relevance = 0.0
    return example


def link_examples_batch(
    examples: list[CodeExample],
    topic_catalog: list[dict],
    api_key: str | None = None,
    min_relevance: float = 0.7,
) -> list[CodeExample]:
    model_id, base_url = resolve_model_for_task("example_linking")
    client = LLMClient(base_url=base_url, model=model_id, api_key=api_key)
    linked: list[CodeExample] = []
    for example in examples:
        link_example_with_llm(example, topic_catalog, client)
        if example.relevance >= min_relevance and example.topic_ids:
            linked.append(example)
    return linked


def load_examples_for_topic(examples: list[CodeExample], topic_id: str) -> list[CodeExample]:
    return [ex for ex in examples if ex.topic_ids and topic_id in ex.topic_ids]
