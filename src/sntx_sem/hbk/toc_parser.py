"""Parser for HBK PackBlock table of contents (1C bracket file)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TocPage:
    block_id: int
    parent_id: int
    child_count: int
    title_ru: str = ""
    title_en: str = ""
    html_path: str = ""
    children: list[TocPage] = field(default_factory=list)

    @property
    def title(self) -> str:
        return self.title_ru or self.title_en

    def walk(self) -> list[TocPage]:
        result = [self]
        for child in self.children:
            result.extend(child.walk())
        return result


def parse_toc(text: str) -> list[TocPage]:
    text = text.lstrip("\ufeff")
    tokens = _tokenize(text)
    pos = 0

    def peek() -> str | None:
        return tokens[pos] if pos < len(tokens) else None

    def consume(expected: str | None = None) -> str:
        nonlocal pos
        tok = tokens[pos]
        pos += 1
        if expected is not None and tok != expected:
            raise ValueError(f"Expected {expected}, got {tok}")
        return tok

    def parse_value() -> str | int | list:
        tok = peek()
        if tok == "{":
            return parse_block()
        if tok == "}":
            raise ValueError("Unexpected }")
        if tok is None:
            raise ValueError("Unexpected end of input")
        consume()
        if tok.lstrip("-").isdigit():
            return int(tok)
        return tok.strip('"')

    def parse_block() -> list:
        consume("{")
        items: list = []
        while peek() != "}":
            if peek() == ",":
                consume(",")
                continue
            items.append(parse_value())
        consume("}")
        return items

    root_items = parse_block()
    if not root_items:
        return []

    blocks_raw = root_items[1:] if isinstance(root_items[0], int) else root_items

    pages: dict[int, TocPage] = {}
    for item in blocks_raw:
        if not isinstance(item, list) or len(item) < 4:
            continue
        block_id = item[0]
        if not isinstance(block_id, int):
            continue
        parent_id = item[1] if isinstance(item[1], int) else 0
        child_count = item[2] if isinstance(item[2], int) else 0
        child_ids: list[int] = []
        meta_idx = 3
        while meta_idx < len(item) and isinstance(item[meta_idx], int):
            child_ids.append(item[meta_idx])
            meta_idx += 1
        title_ru, title_en, html_path = _extract_meta(item[meta_idx:])
        pages[block_id] = TocPage(
            block_id=block_id,
            parent_id=parent_id,
            child_count=child_count,
            title_ru=title_ru,
            title_en=title_en,
            html_path=html_path.replace("\\", "/").lstrip("/"),
        )

    roots: list[TocPage] = []
    for page in pages.values():
        if page.parent_id and page.parent_id in pages:
            pages[page.parent_id].children.append(page)
        elif page.parent_id == 0 or page.parent_id not in pages:
            roots.append(page)

    if not roots and pages:
        roots = [pages[min(pages.keys())]]
    return roots


def _extract_meta(meta: list) -> tuple[str, str, str]:
    title_ru = ""
    title_en = ""
    html_path = ""

    def walk(node) -> None:
        nonlocal title_ru, title_en, html_path
        if isinstance(node, str):
            if (node.startswith("/") or node.endswith(".html") or "/" in node) and not html_path:
                html_path = node
            return
        if not isinstance(node, list):
            return
        if (
            len(node) == 2
            and isinstance(node[0], str)
            and isinstance(node[1], str)
            and node[0] == "#"
        ):
            title_en = node[1]
            return
        if len(node) == 2 and isinstance(node[0], str) and isinstance(node[1], str):
            title_ru = node[0]
            title_en = node[1]
            return
        for child in node:
            walk(child)

    for item in meta:
        walk(item)
        if isinstance(item, str) and item.startswith("/"):
            html_path = item
    return title_ru, title_en, html_path


def _tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch in "{},":
            tokens.append(ch)
            i += 1
            continue
        if ch.isspace():
            i += 1
            continue
        if ch == '"':
            j = i + 1
            buf: list[str] = []
            while j < n:
                if text[j] == '"' and text[j - 1] != "\\":
                    break
                buf.append(text[j])
                j += 1
            tokens.append('"' + "".join(buf) + '"')
            i = j + 1
            continue
        j = i
        while j < n and text[j] not in '{},"' and not text[j].isspace():
            j += 1
        tokens.append(text[i:j])
        i = j
    return tokens


def flatten_toc(roots: list[TocPage]) -> list[TocPage]:
    result: list[TocPage] = []
    for root in roots:
        result.extend(root.walk())
    return result
