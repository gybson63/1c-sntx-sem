"""Hybrid ranking: dense + BM25 fusion and score bonuses."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from rank_bm25 import BM25Okapi

from sntx_sem.index.chunk_text import build_lexical_text
from sntx_sem.index.explain import SearchScoreBreakdown
from sntx_sem.index.tokens import (
    _ACTION_STEMS,
    _EXECUTABLE_KINDS,
    _STATIC_KINDS,
    bm25_tokens,
    looks_like_transformation_query,
    normalize_text,
    query_content_stems,
    semantic_stem_set,
    tokenize_text,
    tokens_related,
)


@dataclass
class SearchFusion:
    scores: dict[str, float]
    breakdowns: dict[str, SearchScoreBreakdown]


@dataclass
class RankerContext:
    """Dependencies required for hybrid ranking over a loaded index."""

    search_config: Any
    chunks: list[dict]
    bm25: BM25Okapi
    get_domain_bm25: Callable[[str], tuple[BM25Okapi | None, list[int]]]
    dense_search: Callable[[list[float], str | None, int], list[dict]]
    known_domains: Callable[[], list[str]]
    count_domain_chunks: Callable[[str], int]


def title_text(chunk: dict) -> str:
    lexical_text = chunk.get("lexical_text", "")
    if isinstance(lexical_text, str) and lexical_text:
        return lexical_text
    return build_lexical_text(chunk)


def title_bonus(query: str, chunk: dict) -> float:
    query_tokens = tokenize_text(query)
    if not query_tokens:
        return 0.0

    title_tokens = tokenize_text(title_text(chunk))
    if not title_tokens:
        return 0.0

    matched_tokens = sum(
        1
        for query_token in query_tokens
        if any(tokens_related(query_token, title_token) for title_token in title_tokens)
    )
    coverage = matched_tokens / len(query_tokens)

    normalized_query = normalize_text(query)
    normalized_title = normalize_text(" ".join(title_tokens))
    is_name_query = len(query_content_stems(query)) <= 1
    if is_name_query:
        phrase_bonus = 0.03 if normalized_query and normalized_query in normalized_title else 0.0
        return (coverage * 0.02) + phrase_bonus
    phrase_bonus = 0.006 if normalized_query and normalized_query in normalized_title else 0.0
    return (coverage * 0.004) + phrase_bonus


def semantic_intent_bonus(query: str, chunk: dict) -> float:
    query_stems = query_content_stems(query)
    if len(query_stems) < 2:
        return 0.0
    semantic_stems = semantic_stem_set(chunk)
    if not semantic_stems:
        return 0.0

    matched = sum(1 for stem in query_stems if stem in semantic_stems)
    if matched < 2:
        return 0.0

    transformation_intent = looks_like_transformation_query(query)
    kind = str(chunk.get("entity_kind", "")).lower()
    domain = str(chunk.get("domain", "")).lower()

    has_action = any(
        any(stem.startswith(action_stem) for action_stem in _ACTION_STEMS)
        for stem in semantic_stems
    )

    bonus = 0.0
    if kind in _EXECUTABLE_KINDS and has_action:
        bonus += 0.06
        if transformation_intent and domain == "platform_api":
            bonus += 0.04
    elif transformation_intent and kind in _STATIC_KINDS:
        bonus -= 0.02
    return bonus


def fuse_dense_bm25(
    ctx: RankerContext,
    query: str,
    query_vector: list[float],
    domain_filter: str | None,
    candidate_limit: int,
) -> SearchFusion:
    dense_limit = max(ctx.search_config.dense_top_k, candidate_limit)
    dense_hits = ctx.dense_search(query_vector, domain_filter, dense_limit)

    rrf_k = ctx.search_config.rrf_k
    dense_weight = float(getattr(ctx.search_config, "dense_rrf_weight", 1.35))
    bm25_weight = float(getattr(ctx.search_config, "bm25_rrf_weight", 1.0))
    dense_similarity_weight = float(getattr(ctx.search_config, "dense_similarity_weight", 1.0))
    scores: dict[str, float] = {}
    breakdowns: dict[str, SearchScoreBreakdown] = {}

    for rank, row in enumerate(dense_hits, 1):
        cid = row["id"]
        dense_rank_score = dense_weight / (rrf_k + rank)
        distance = float(row.get("_distance", 1.0))
        dense_similarity_score = dense_similarity_weight / (1.0 + max(distance, 0.0))
        scores[cid] = scores.get(cid, 0) + dense_rank_score + dense_similarity_score
        breakdown = breakdowns.setdefault(cid, SearchScoreBreakdown())
        breakdown.dense_rank = rank
        breakdown.dense_distance = distance
        breakdown.dense_rrf += dense_rank_score
        breakdown.dense_similarity += dense_similarity_score

    if domain_filter:
        bm25, chunk_indices = ctx.get_domain_bm25(domain_filter)
        if bm25 is not None:
            bm25_scores = bm25.get_scores(bm25_tokens(query))
            bm25_ranked = sorted(
                enumerate(bm25_scores),
                key=lambda x: x[1],
                reverse=True,
            )[: ctx.search_config.bm25_top_k]
            for rank, (idx, raw_score) in enumerate(bm25_ranked, 1):
                chunk = ctx.chunks[chunk_indices[idx]]
                cid = chunk["id"]
                bm25_rank_score = bm25_weight / (rrf_k + rank)
                scores[cid] = scores.get(cid, 0) + bm25_rank_score
                breakdown = breakdowns.setdefault(cid, SearchScoreBreakdown())
                breakdown.bm25_rank = rank
                breakdown.bm25_raw = float(raw_score)
                breakdown.bm25_rrf += bm25_rank_score
    else:
        bm25_scores = ctx.bm25.get_scores(bm25_tokens(query))
        bm25_ranked = sorted(
            enumerate(bm25_scores),
            key=lambda x: x[1],
            reverse=True,
        )[: ctx.search_config.bm25_top_k]
        for rank, (idx, raw_score) in enumerate(bm25_ranked, 1):
            chunk = ctx.chunks[idx]
            cid = chunk["id"]
            bm25_rank_score = bm25_weight / (rrf_k + rank)
            scores[cid] = scores.get(cid, 0) + bm25_rank_score
            breakdown = breakdowns.setdefault(cid, SearchScoreBreakdown())
            breakdown.bm25_rank = rank
            breakdown.bm25_raw = float(raw_score)
            breakdown.bm25_rrf += bm25_rank_score

    natural_max = ctx.search_config.dense_top_k + ctx.search_config.bm25_top_k
    if len(scores) <= candidate_limit or len(scores) <= natural_max:
        return SearchFusion(scores=scores, breakdowns=breakdowns)
    top_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:candidate_limit]
    return SearchFusion(
        scores={cid: scores[cid] for cid in top_ids},
        breakdowns={cid: breakdowns[cid] for cid in top_ids if cid in breakdowns},
    )


def fuse_bm25_only(
    ctx: RankerContext,
    query: str,
    domain_filter: str,
    candidate_limit: int,
) -> SearchFusion:
    rrf_k = ctx.search_config.rrf_k
    bm25_weight = float(getattr(ctx.search_config, "bm25_rrf_weight", 1.0))
    scores: dict[str, float] = {}
    breakdowns: dict[str, SearchScoreBreakdown] = {}

    bm25, chunk_indices = ctx.get_domain_bm25(domain_filter)
    if bm25 is not None:
        bm25_limit = max(ctx.search_config.bm25_top_k, candidate_limit)
        bm25_scores = bm25.get_scores(bm25_tokens(query))
        bm25_ranked = sorted(
            enumerate(bm25_scores),
            key=lambda x: x[1],
            reverse=True,
        )[:bm25_limit]
        for rank, (idx, raw_score) in enumerate(bm25_ranked, 1):
            chunk = ctx.chunks[chunk_indices[idx]]
            cid = chunk["id"]
            bm25_rank_score = bm25_weight / (rrf_k + rank)
            scores[cid] = scores.get(cid, 0) + bm25_rank_score
            breakdown = breakdowns.setdefault(cid, SearchScoreBreakdown())
            breakdown.bm25_rank = rank
            breakdown.bm25_raw = float(raw_score)
            breakdown.bm25_rrf += bm25_rank_score

    if len(scores) <= candidate_limit:
        return SearchFusion(scores=scores, breakdowns=breakdowns)
    top_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:candidate_limit]
    return SearchFusion(
        scores={cid: scores[cid] for cid in top_ids},
        breakdowns={cid: breakdowns[cid] for cid in top_ids if cid in breakdowns},
    )


def per_domain_candidate_limit(ctx: RankerContext, domain: str, final_k: int) -> int:
    domain_size = ctx.count_domain_chunks(domain)
    if domain_size == 0:
        return 0
    scaled = max(final_k * 4, int(domain_size**0.5) * 2)
    minimum = max(ctx.search_config.dense_top_k, ctx.search_config.bm25_top_k)
    return int(min(domain_size, max(minimum, scaled)))


def all_domain_candidates(
    ctx: RankerContext,
    query: str,
    query_vector: list[float],
    final_k: int,
) -> SearchFusion:
    candidate_limit = max(
        final_k * 8,
        ctx.search_config.dense_top_k,
        ctx.search_config.bm25_top_k,
    )
    merged = fuse_dense_bm25(ctx, query, query_vector, None, candidate_limit)

    for domain in ctx.known_domains():
        per_domain_limit = per_domain_candidate_limit(ctx, domain, final_k)
        if per_domain_limit <= 0:
            continue
        domain_scores = fuse_bm25_only(ctx, query, domain, per_domain_limit)
        for cid, score in domain_scores.scores.items():
            prev = merged.scores.get(cid)
            if prev is None or score > prev:
                merged.scores[cid] = score
                if cid in domain_scores.breakdowns:
                    merged.breakdowns[cid] = domain_scores.breakdowns[cid]

    return merged


def apply_title_bonus(
    query: str,
    fusion: SearchFusion,
    chunk_map: dict[str, dict],
) -> SearchFusion:
    boosted: dict[str, float] = {}
    for cid, score in fusion.scores.items():
        chunk = chunk_map.get(cid)
        if not chunk:
            continue
        title_bonus_value = title_bonus(query, chunk)
        semantic_intent_bonus_value = semantic_intent_bonus(query, chunk)
        total = score + title_bonus_value + semantic_intent_bonus_value
        boosted[cid] = total
        breakdown = fusion.breakdowns.setdefault(cid, SearchScoreBreakdown())
        breakdown.title_bonus = title_bonus_value
        breakdown.semantic_intent_bonus = semantic_intent_bonus_value
        breakdown.total = total
    return SearchFusion(scores=boosted, breakdowns=fusion.breakdowns)
