"""Tokenization and query normalization for hybrid search."""

from __future__ import annotations

import re

_IDENTIFIER_WORD_RE = re.compile(r"[А-ЯЁA-Z][а-яёa-z]*")
_QUERY_TOKEN_RE = re.compile(r"[a-zа-яё0-9]+", re.IGNORECASE)

_RU_SUFFIXES = (
    "ами",
    "ями",
    "ого",
    "ему",
    "ими",
    "ому",
    "ах",
    "ях",
    "ов",
    "ев",
    "ей",
    "ий",
    "ый",
    "ая",
    "ое",
    "ые",
    "у",
    "а",
    "е",
    "и",
    "о",
    "ы",
    "ь",
    "ю",
    "я",
)

_QUERY_STOPWORDS = {"и", "в", "во", "на", "по", "к", "с", "со", "из", "для", "о", "об", "от"}
_EXECUTABLE_KINDS = {"method", "function", "constructor", "procedure"}
_STATIC_KINDS = {"property", "type", "enum", "topic", "page"}
_ACTION_STEMS = (
    "раздел",
    "разбив",
    "преобраз",
    "конверт",
    "соедин",
    "подстав",
    "объедин",
    "split",
    "concat",
)


def split_identifier(name: str) -> str:
    """Split 1C CamelCase identifier (Cyrillic/Latin) into spaced lowercase words."""
    words: list[str] = []
    for segment in re.split(r"[.\s]+", name.strip()):
        if not segment:
            continue
        parts = _IDENTIFIER_WORD_RE.findall(segment)
        if parts:
            words.extend(parts)
        else:
            words.append(segment)
    return " ".join(word.lower() for word in words)


def normalize_text(text: str) -> str:
    return " ".join(_QUERY_TOKEN_RE.findall(text.lower()))


def tokenize_text(text: str) -> list[str]:
    return _QUERY_TOKEN_RE.findall(text.lower())


def token_stem(token: str) -> str:
    for suffix in _RU_SUFFIXES:
        if len(token) > len(suffix) + 2 and token.endswith(suffix):
            return token[: -len(suffix)]
    return token


def tokens_related(query_token: str, title_token: str) -> bool:
    if query_token == title_token:
        return True
    if (
        len(query_token) >= 3
        and len(title_token) >= 3
        and (query_token.startswith(title_token) or title_token.startswith(query_token))
    ):
        return True
    query_stem = token_stem(query_token)
    title_stem = token_stem(title_token)
    if query_stem == title_stem:
        return True
    return (
        len(query_stem) >= 3
        and len(title_stem) >= 3
        and (query_stem.startswith(title_stem) or title_stem.startswith(query_stem))
    )


def query_content_stems(query: str) -> list[str]:
    stems: list[str] = []
    for token in tokenize_text(query):
        stem = token_stem(token)
        if len(stem) < 3:
            continue
        if stem in _QUERY_STOPWORDS:
            continue
        if stem not in stems:
            stems.append(stem)
    return stems


def semantic_stem_set(chunk: dict) -> set[str]:
    semantic_text = chunk.get("semantic_text", "")
    if not semantic_text:
        semantic_text = chunk.get("text", "")
    return {token_stem(token) for token in tokenize_text(semantic_text)}


def looks_like_transformation_query(query: str) -> bool:
    tokens = tokenize_text(query)
    return "в" in tokens and len(query_content_stems(query)) >= 2


def semantic_query_text(query: str) -> str:
    if not looks_like_transformation_query(query):
        return query
    tokens = tokenize_text(query)
    content_tokens = [
        token
        for token in tokens
        if len(token_stem(token)) >= 3 and token_stem(token) not in _QUERY_STOPWORDS
    ]
    if len(content_tokens) < 2:
        return query
    source = content_tokens[0]
    target = content_tokens[-1]
    expanded = [
        query,
        f"преобразовать {source} в {target}",
        f"получить {target} из {source}",
    ]
    source_stem = token_stem(source)
    target_stem = token_stem(target)
    if source_stem.startswith("строк") and target_stem.startswith("масс"):
        expanded.append("разделить строку")
    return ". ".join(expanded)


def bm25_tokens(text: str) -> list[str]:
    return [token_stem(token) for token in tokenize_text(text)]
