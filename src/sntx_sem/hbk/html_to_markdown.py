"""Convert HBK HTML help pages to plain text / markdown."""

from __future__ import annotations

import re

from bs4 import BeautifulSoup, NavigableString, Tag


def html_to_markdown(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "meta", "link"]):
        tag.decompose()
    body = soup.body or soup
    return _render(body).strip()


def _render(node: Tag | NavigableString) -> str:
    if isinstance(node, NavigableString):
        return re.sub(r"\s+", " ", str(node))
    if not isinstance(node, Tag):
        return ""

    name = node.name.lower()
    if name in {"h1", "h2", "h3", "h4"}:
        level = int(name[1])
        text = _inline(node)
        return f"\n{'#' * level} {text}\n"
    if name == "p":
        return f"\n{_inline(node)}\n"
    if name == "pre":
        return f"\n```\n{node.get_text()}\n```\n"
    if name == "table":
        return _render_table(node)
    if name == "ul":
        return "\n".join(f"- {_inline(li)}" for li in node.find_all("li", recursive=False))
    if name == "ol":
        lines = []
        for i, li in enumerate(node.find_all("li", recursive=False), 1):
            lines.append(f"{i}. {_inline(li)}")
        return "\n".join(lines)
    if name == "br":
        return "\n"
    return "".join(
        _render(child) for child in node.children if isinstance(child, (Tag, NavigableString))
    )


def _inline(node: Tag) -> str:
    parts: list[str] = []
    for child in node.children:
        if isinstance(child, NavigableString):
            parts.append(str(child))
        elif isinstance(child, Tag):
            if child.name in {"b", "strong"}:
                parts.append(f"**{child.get_text(strip=True)}**")
            elif child.name in {"i", "em"}:
                parts.append(f"*{child.get_text(strip=True)}*")
            elif child.name == "a":
                parts.append(child.get_text(strip=True))
            else:
                parts.append(child.get_text(" ", strip=True))
    return re.sub(r"\s+", " ", "".join(parts)).strip()


def _render_table(table: Tag) -> str:
    rows = table.find_all("tr")
    if not rows:
        return ""
    lines: list[str] = []
    for i, row in enumerate(rows):
        cells = [c.get_text(" ", strip=True) for c in row.find_all(["td", "th"])]
        lines.append("| " + " | ".join(cells) + " |")
        if i == 0:
            lines.append("| " + " | ".join("---" for _ in cells) + " |")
    return "\n" + "\n".join(lines) + "\n"


def extract_sections(markdown: str) -> dict[str, str]:
    """Split help page into named sections."""
    sections: dict[str, str] = {"body": markdown}
    current = "body"
    buf: list[str] = []
    for line in markdown.splitlines():
        if line.startswith("#"):
            if buf:
                sections[current] = "\n".join(buf).strip()
                buf = []
            title = line.lstrip("#").strip().lower()
            if "syntax" in title or "синтаксис" in title:
                current = "syntax"
            elif "param" in title or "параметр" in title:
                current = "parameters"
            elif "example" in title or "пример" in title:
                current = "example"
            elif "description" in title or "описание" in title:
                current = "description"
            else:
                current = title.replace(" ", "_")
            buf.append(line)
        else:
            buf.append(line)
    if buf:
        sections[current] = "\n".join(buf).strip()
    return sections
