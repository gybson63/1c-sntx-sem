"""Example store backed by JSONL (linked to help topics)."""

from __future__ import annotations

import json
from pathlib import Path

from sntx_sem.examples.scanner import CodeExample


class ExamplesStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._examples: list[CodeExample] = []
        if path.is_file():
            self.load()

    def load(self) -> None:
        self._examples = []
        with self.path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    self._examples.append(CodeExample(**data))

    def save(self, examples: list[CodeExample]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as f:
            for ex in examples:
                f.write(json.dumps(ex.to_dict(), ensure_ascii=False) + "\n")
        self._examples = examples

    def find_by_topic(self, topic_id: str, limit: int = 5) -> list[CodeExample]:
        matched = [ex for ex in self._examples if ex.topic_ids and topic_id in ex.topic_ids]
        matched.sort(key=lambda x: x.relevance, reverse=True)
        return matched[:limit]

    def search(self, query: str, limit: int = 5) -> list[CodeExample]:
        q = query.lower()
        scored: list[tuple[float, CodeExample]] = []
        for ex in self._examples:
            text = f"{ex.code} {ex.query_text} {ex.summary}".lower()
            if q in text:
                scored.append((1.0, ex))
                continue
            words = [w for w in q.split() if len(w) > 2]
            if not words:
                continue
            overlap = sum(1 for w in words if w in text) / len(words)
            if overlap > 0.3:
                scored.append((overlap, ex))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [ex for _, ex in scored[:limit]]

    @property
    def count(self) -> int:
        return len(self._examples)
