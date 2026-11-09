"""Example store backed by JSONL (linked to help topics)."""

from __future__ import annotations

import json
from pathlib import Path

from sntx_sem.examples.scanner import CodeExample, write_text_atomic


def _token_overlap(query: str, text: str) -> float:
    words = [word for word in query.split() if len(word) > 2]
    if not words:
        return 0.0
    return sum(1 for word in words if word in text) / len(words)


def score_example(query: str, example: CodeExample) -> float | None:
    """Rank formulated questions above a raw code substring."""
    needle = query.lower().strip()
    if not needle:
        return None
    question_text = " ".join(
        [*(example.questions or []), example.task or "", example.summary or ""]
    ).lower()
    body = f"{example.code} {example.query_text}".lower()
    question_overlap = _token_overlap(needle, question_text)
    body_overlap = _token_overlap(needle, body)
    if question_text and needle in question_text:
        return 2.0 + question_overlap
    if question_overlap > 0:
        return 1.0 + question_overlap
    if needle in body:
        return 0.5
    if body_overlap > 0.3:
        return body_overlap * 0.4
    return None


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
                    self._examples.append(CodeExample.from_dict(data))

    def save(self, examples: list[CodeExample]) -> None:
        payload = "".join(json.dumps(ex.to_dict(), ensure_ascii=False) + "\n" for ex in examples)
        write_text_atomic(self.path, payload)
        self._examples = examples

    def find_by_topic(self, topic_id: str, limit: int = 5) -> list[CodeExample]:
        matched = [ex for ex in self._examples if ex.topic_ids and topic_id in ex.topic_ids]
        matched.sort(key=lambda x: x.relevance, reverse=True)
        return matched[:limit]

    def search(self, query: str, limit: int = 5) -> list[CodeExample]:
        scored: list[tuple[float, CodeExample]] = []
        for ex in self._examples:
            score = score_example(query, ex)
            if score is None:
                continue
            scored.append((score, ex))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [ex for _, ex in scored[:limit]]

    @property
    def count(self) -> int:
        return len(self._examples)
