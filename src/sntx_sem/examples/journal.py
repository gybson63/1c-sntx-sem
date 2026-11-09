"""Persistent history of mined procedures across local configurations."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sntx_sem.examples.procedures import ProcedureUnit

PENDING = "pending"
DONE = "done"
FAILED = "failed"
REMOVED = "removed"
TRACKED_STATUSES = (PENDING, DONE, FAILED)


@dataclass
class JournalEntry:
    config_label: str
    module_path: str
    procedure: str
    start_line: int
    end_line: int
    content_hash: str
    status: str
    updated_at: str
    examples: list[dict[str, Any]] = field(default_factory=list)
    skip_reason: str = ""
    error: str = ""
    config_index: int = 0
    module_rank: int = 0

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.config_label, self.module_path, self.procedure)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> JournalEntry:
        known = {item.name for item in fields(cls)}
        payload = {key: value for key, value in data.items() if key in known}
        return cls(**payload)


class MineJournal:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.entries: dict[tuple[str, str, str], JournalEntry] = {}

    @classmethod
    def load(cls, path: Path) -> MineJournal:
        journal = cls(path)
        if not path.is_file():
            return journal
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                entry = JournalEntry.from_dict(json.loads(line))
                journal.entries[entry.key] = entry
        return journal

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        ordered = sorted(self.entries.values(), key=_entry_sort_key)
        payload = "".join(
            json.dumps(entry.to_dict(), ensure_ascii=False) + "\n" for entry in ordered
        )
        temporary = self.path.with_suffix(".jsonl.tmp")
        temporary.write_text(payload, encoding="utf-8")
        temporary.replace(self.path)

    def sync(self, units: list[ProcedureUnit], available: set[str]) -> None:
        """Reconcile the journal with modules that are on disk right now."""
        seen: set[tuple[str, str, str]] = set()
        now = _now()
        for unit in units:
            seen.add(unit.key)
            current = self.entries.get(unit.key)
            if current is None:
                self.entries[unit.key] = JournalEntry(
                    config_label=unit.config_label,
                    module_path=unit.module_path,
                    procedure=unit.procedure,
                    start_line=unit.start_line,
                    end_line=unit.end_line,
                    content_hash=unit.content_hash,
                    status=PENDING,
                    updated_at=now,
                    config_index=unit.config_index,
                    module_rank=unit.module_rank,
                )
                continue
            current.config_index = unit.config_index
            current.module_rank = unit.module_rank
            current.start_line = unit.start_line
            current.end_line = unit.end_line
            reset = current.content_hash != unit.content_hash or current.status == REMOVED
            current.content_hash = unit.content_hash
            if reset:
                current.status = PENDING
                current.examples = []
                current.skip_reason = ""
                current.error = ""
                current.updated_at = now
        for entry in self.entries.values():
            if entry.config_label not in available:
                continue
            if entry.key in seen or entry.status == REMOVED:
                continue
            entry.status = REMOVED
            entry.examples = []
            entry.skip_reason = ""
            entry.error = ""
            entry.updated_at = now

    def select(
        self,
        *,
        limit: int | None,
        labels: set[str],
        include: str,
        retry_failed: bool,
        available: set[str],
    ) -> list[JournalEntry]:
        status = FAILED if retry_failed else PENDING
        chosen: list[JournalEntry] = []
        for entry in sorted(self.entries.values(), key=_entry_sort_key):
            if entry.status != status or entry.config_label not in available:
                continue
            if labels and entry.config_label not in labels:
                continue
            if include and not _path_matches(entry.module_path, include):
                continue
            chosen.append(entry)
            if limit is not None and len(chosen) >= limit:
                break
        return chosen

    def next_pending(self, available: set[str]) -> JournalEntry | None:
        batch = self.select(
            limit=1,
            labels=set(),
            include="",
            retry_failed=False,
            available=available,
        )
        return batch[0] if batch else None

    def counts_by_label(self) -> list[tuple[str, dict[str, int]]]:
        grouped: dict[str, dict[str, int]] = {}
        order: dict[str, int] = {}
        for entry in self.entries.values():
            bucket = grouped.setdefault(entry.config_label, {name: 0 for name in TRACKED_STATUSES})
            order.setdefault(entry.config_label, entry.config_index)
            if entry.status in bucket:
                bucket[entry.status] += 1
        return [(label, grouped[label]) for label in sorted(grouped, key=lambda item: order[item])]


def format_status(journal: MineJournal, available: set[str]) -> str:
    lines: list[str] = []
    grouped = journal.counts_by_label()
    if not grouped:
        lines.append("журнал пуст")
    else:
        totals = {name: 0 for name in TRACKED_STATUSES}
        detail: list[str] = []
        for label, counts in grouped:
            for name in TRACKED_STATUSES:
                totals[name] += counts[name]
            detail.append(
                f"{label}: pending={counts[PENDING]} done={counts[DONE]} failed={counts[FAILED]}"
            )
        lines.append(
            f"итого: pending={totals[PENDING]} done={totals[DONE]} failed={totals[FAILED]}"
        )
        lines.extend(detail)
    nxt = journal.next_pending(available)
    if nxt is None:
        lines.append("следующая: нет")
    else:
        lines.append(f"следующая: {nxt.config_label} {nxt.module_path} :: {nxt.procedure}")
    return "\n".join(lines)


def _entry_sort_key(entry: JournalEntry) -> tuple[int, int, str, int, str]:
    return (
        entry.config_index,
        entry.module_rank,
        entry.module_path.lower(),
        entry.start_line,
        entry.procedure,
    )


def _path_matches(module_path: str, include: str) -> bool:
    needle = include.replace("\\", "/").lower()
    return needle in module_path.replace("\\", "/").lower()


def _now() -> str:
    return datetime.now(UTC).isoformat()
