"""Search result explainability: highlights, excerpts, match sources."""

from __future__ import annotations

from dataclasses import dataclass, field

from sntx_sem.index.tokens import (
    _ACTION_STEMS,
    looks_like_transformation_query,
    query_content_stems,
    token_stem,
    tokenize_text,
)


@dataclass
class SearchScoreBreakdown:
    dense_rrf: float = 0.0
    dense_similarity: float = 0.0
    bm25_rrf: float = 0.0
    title_bonus: float = 0.0
    semantic_intent_bonus: float = 0.0
    dense_rank: int | None = None
    bm25_rank: int | None = None
    dense_distance: float | None = None
    bm25_raw: float | None = None
    total: float = 0.0

    def to_dict(self) -> dict[str, float | int | None]:
        return {
            "total": self.total,
            "dense_rrf": self.dense_rrf,
            "dense_similarity": self.dense_similarity,
            "bm25_rrf": self.bm25_rrf,
            "title_bonus": self.title_bonus,
            "semantic_intent_bonus": self.semantic_intent_bonus,
            "dense_rank": self.dense_rank,
            "bm25_rank": self.bm25_rank,
            "dense_distance": self.dense_distance,
            "bm25_raw": self.bm25_raw,
        }


@dataclass
class SearchResult:
    id: str
    domain: str
    title: str
    text: str
    score: float
    entity_kind: str = ""
    html_path: str = ""
    syntax: str = ""
    highlight_terms: list[str] = field(default_factory=list)
    match_sources: list[str] = field(default_factory=list)
    match_explanation: str = ""
    semantic_excerpt: str = ""
    semantic_highlight_terms: list[str] = field(default_factory=list)
    score_breakdown: dict[str, float | int | None] = field(default_factory=dict)


def collect_matched_terms(query: str, search_text: str) -> list[str]:
    query_tokens = tokenize_text(query)
    if not query_tokens or not search_text:
        return []
    search_tokens = set(tokenize_text(search_text))
    matched: list[str] = []
    for token in query_tokens:
        if token in search_tokens and token not in matched:
            matched.append(token)
    return matched


def semantic_text_for_explanation(chunk: dict) -> str:
    semantic_text = chunk.get("semantic_text", "")
    if isinstance(semantic_text, str) and semantic_text:
        return semantic_text
    text = chunk.get("text", "")
    return text if isinstance(text, str) else ""


def collect_semantic_highlight_terms(query: str, chunk: dict) -> list[str]:
    query_stems = query_content_stems(query)
    semantic_text = semantic_text_for_explanation(chunk)
    if not query_stems or not semantic_text:
        return []

    terms: list[str] = []
    transformation_intent = looks_like_transformation_query(query)
    for token in tokenize_text(semantic_text):
        stem = token_stem(token)
        if len(stem) < 3:
            continue
        matches_query = any(
            stem == query_stem or stem.startswith(query_stem) or query_stem.startswith(stem)
            for query_stem in query_stems
        )
        matches_intent = transformation_intent and any(
            stem.startswith(action_stem) for action_stem in _ACTION_STEMS
        )
        if (matches_query or matches_intent) and token not in terms:
            terms.append(token)
        if len(terms) >= 8:
            break
    return terms


def find_first_term_position(text: str, terms: list[str]) -> int | None:
    lowered = text.lower()
    positions = [lowered.find(term.lower()) for term in terms if term]
    found = [position for position in positions if position >= 0]
    return min(found) if found else None


def build_semantic_excerpt(text: str, terms: list[str], max_chars: int = 360) -> str:
    if not text:
        return ""
    if len(text) <= max_chars:
        return text

    match_pos = find_first_term_position(text, terms)
    if match_pos is None:
        return text[:max_chars] + "..."

    half_window = max_chars // 2
    start = max(0, match_pos - half_window)
    end = min(len(text), start + max_chars)
    start = max(0, end - max_chars)
    snippet = text[start:end]
    if start > 0:
        snippet = "..." + snippet
    if end < len(text):
        snippet = snippet + "..."
    return snippet


def match_sources(breakdown: SearchScoreBreakdown) -> list[str]:
    sources: list[str] = []
    if breakdown.dense_rrf > 0 or breakdown.dense_similarity > 0:
        sources.append("semantic")
    if breakdown.bm25_rrf > 0:
        sources.append("bm25")
    if breakdown.title_bonus > 0:
        sources.append("title")
    if breakdown.semantic_intent_bonus > 0:
        sources.append("intent")
    return sources


def match_explanation(breakdown: SearchScoreBreakdown) -> str:
    parts: list[str] = []
    if breakdown.dense_rank is not None:
        parts.append(f"Семантический поиск: rank {breakdown.dense_rank}")
    if breakdown.bm25_rank is not None:
        parts.append(f"BM25: rank {breakdown.bm25_rank}")
    if breakdown.title_bonus > 0:
        parts.append("совпадение с названием добавило бонус")
    if breakdown.semantic_intent_bonus > 0:
        parts.append("описание совпало с намерением запроса")
    if not parts:
        return "Результат попал в выдачу по суммарному гибридному score."
    return "; ".join(parts) + "."


def build_search_results(
    query: str,
    scores: dict[str, float],
    breakdowns: dict[str, SearchScoreBreakdown],
    chunk_map: dict[str, dict],
) -> list[SearchResult]:
    ranked_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    results: list[SearchResult] = []
    for cid in ranked_ids:
        chunk = chunk_map.get(cid)
        if not chunk:
            continue
        title = chunk.get("title_ru") or chunk.get("title_en") or chunk.get("id", "")
        breakdown = breakdowns.get(cid, SearchScoreBreakdown(total=scores[cid]))
        if not breakdown.total:
            breakdown.total = scores[cid]
        semantic_terms = collect_semantic_highlight_terms(query, chunk)
        semantic_text = semantic_text_for_explanation(chunk)
        results.append(
            SearchResult(
                id=cid,
                domain=chunk.get("domain", ""),
                title=title,
                text=chunk.get("text", ""),
                score=scores[cid],
                entity_kind=chunk.get("entity_kind", ""),
                html_path=chunk.get("html_path", ""),
                syntax=chunk.get("syntax", ""),
                highlight_terms=collect_matched_terms(query, chunk.get("search_text", "")),
                match_sources=match_sources(breakdown),
                match_explanation=match_explanation(breakdown),
                semantic_excerpt=build_semantic_excerpt(semantic_text, semantic_terms),
                semantic_highlight_terms=semantic_terms,
                score_breakdown=breakdown.to_dict(),
            )
        )
    return results
