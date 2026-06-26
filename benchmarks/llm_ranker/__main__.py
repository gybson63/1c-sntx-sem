"""LLM ranking benchmark for 1C help tasks."""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import click
import yaml

from sntx_sem.llm.client import LLMClient, parse_json_response


@dataclass
class ModelCandidate:
    key: str
    model_id: str
    base_url: str
    api_key_env: str = "OPENAI_API_KEY"


@dataclass
class TrackResult:
    model_key: str
    accuracy: float
    json_validity: float
    avg_latency_ms: float
    estimated_cost: float
    score: float


DEFAULT_CANDIDATES = [
    ModelCandidate(
        "deepseek-chat", "deepseek-chat", "https://api.deepseek.com/v1", "DEEPSEEK_API_KEY"
    ),
    ModelCandidate("gpt-4o-mini", "gpt-4o-mini", "https://api.openai.com/v1", "OPENAI_API_KEY"),
    ModelCandidate(
        "qwen-coder",
        "qwen2.5-coder-32b-instruct",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "DASHSCOPE_API_KEY",
    ),
    ModelCandidate(
        "gemini-flash",
        "gemini-2.0-flash",
        "https://generativelanguage.googleapis.com/v1beta/openai",
        "GEMINI_API_KEY",
    ),
]


def load_dataset(name: str, datasets_dir: Path) -> list[dict]:
    path = datasets_dir / f"{name}.jsonl"
    if not path.is_file():
        return []
    items = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


def mrr_at_k(ranked: list[str], expected: list[str], k: int = 5) -> float:
    for rank, item in enumerate(ranked[:k], 1):
        if item in expected:
            return 1.0 / rank
    return 0.0


def run_retrieval_qa(
    client: LLMClient,
    dataset: list[dict],
    index_search_fn,
) -> tuple[float, float]:
    """Evaluate retrieval using index search (not LLM). Returns MRR, latency."""
    if not dataset:
        return 0.0, 0.0
    scores = []
    latencies = []
    for item in dataset:
        start = time.perf_counter()
        results = index_search_fn(item["query"], item.get("domain", "all"), 5)
        latencies.append((time.perf_counter() - start) * 1000)
        ranked_ids = [r.id for r in results]
        scores.append(mrr_at_k(ranked_ids, item["expected_topic_ids"]))
    return statistics.mean(scores), statistics.mean(latencies)


def run_example_linking(client: LLMClient, dataset: list[dict]) -> tuple[float, float, float]:
    if not dataset:
        return 0.0, 1.0, 0.0
    correct = 0
    valid_json = 0
    latencies = []
    for item in dataset:
        messages = [
            {
                "role": "user",
                "content": (
                    "Match this 1C code to topic IDs from "
                    f"{json.dumps(item['topic_options'], ensure_ascii=False)}. "
                    f"Code:\n```\n{item['code']}\n```\n"
                    'Return JSON: {"topic_ids": ["..."], "relevance": 0.9}'
                ),
            }
        ]
        start = time.perf_counter()
        try:
            text = client.chat(messages)
            result = parse_json_response(text)
            valid_json += 1
            predicted = set(result.get("topic_ids", []))
            expected = set(item["expected_topic_ids"])
            if predicted & expected:
                correct += 1
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
        latencies.append((time.perf_counter() - start) * 1000)
    n = len(dataset)
    return correct / n, valid_json / n, statistics.mean(latencies)


def run_reranking(client: LLMClient, dataset: list[dict]) -> tuple[float, float]:
    if not dataset:
        return 0.0, 0.0
    taus = []
    latencies = []
    for item in dataset:
        messages = [
            {
                "role": "user",
                "content": (
                    f"Query: {item['query']}\n"
                    f"Candidates: {json.dumps(item['candidates'], ensure_ascii=False)}\n"
                    'Return JSON: {"ranked_ids": ["id1", "id2", ...]}'
                ),
            }
        ]
        start = time.perf_counter()
        try:
            result = parse_json_response(client.chat(messages))
            ranked = result.get("ranked_ids", [])
            gold = item["gold_ranking"]
            common = [x for x in ranked if x in gold]
            if len(common) >= 2:
                tau = sum(
                    1
                    for i in range(len(common) - 1)
                    if gold.index(common[i]) < gold.index(common[i + 1])
                ) / max(len(common) - 1, 1)
            else:
                tau = 1.0 if ranked and ranked[0] == gold[0] else 0.0
            taus.append(tau)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            taus.append(0.0)
        latencies.append((time.perf_counter() - start) * 1000)
    return statistics.mean(taus), statistics.mean(latencies)


def composite_score(
    accuracy: float,
    json_validity: float,
    latency_ms: float,
    cost: float,
    max_latency: float = 5000,
    max_cost: float = 0.01,
) -> float:
    norm_lat = min(latency_ms / max_latency, 1.0)
    norm_cost = min(cost / max_cost, 1.0)
    return 0.45 * accuracy + 0.25 * json_validity + 0.15 * (1 - norm_cost) + 0.15 * (1 - norm_lat)


@click.group()
def main() -> None:
    """LLM ranking benchmark."""


@main.command("run")
@click.option("--datasets-dir", type=click.Path(exists=True), default=None)
@click.option("--output", type=click.Path(), default="config/benchmark_results.yaml")
@click.option("--skip-llm", is_flag=True, help="Skip LLM tracks (retrieval only)")
def run_benchmark(datasets_dir: str | None, output: str, skip_llm: bool) -> None:
    """Run benchmark and write winners to benchmark_results.yaml."""
    import os

    base = Path(__file__).resolve().parent.parent
    ds_dir = Path(datasets_dir) if datasets_dir else base / "datasets"
    out_path = Path(output)

    retrieval_ds = load_dataset("retrieval_qa", ds_dir)
    linking_ds = load_dataset("example_linking", ds_dir)
    rerank_ds = load_dataset("reranking", ds_dir)

    index_search_fn = None
    if retrieval_ds:
        try:
            from sntx_sem.config import load_config
            from sntx_sem.embeddings import create_embedding_backend
            from sntx_sem.index.store import HelpIndex

            cfg = load_config()
            backend = create_embedding_backend(cfg.embedding)
            idx = HelpIndex(cfg.index_dir, backend, cfg.search)

            def index_search_fn(query: str, domain: str, limit: int):
                return idx.search(query, domain=domain, limit=limit)

            mrr, lat = run_retrieval_qa(None, retrieval_ds, index_search_fn)
            click.echo(f"Retrieval QA (index): MRR@5={mrr:.3f}, latency={lat:.0f}ms")
        except Exception as exc:
            click.echo(f"Retrieval QA skipped: {exc}")

    results: dict[str, TrackResult] = {}
    for cand in DEFAULT_CANDIDATES:
        api_key = os.environ.get(cand.api_key_env)
        if skip_llm or not api_key:
            continue
        client = LLMClient(cand.base_url, cand.model_id, api_key)
        acc, json_val, lat = run_example_linking(client, linking_ds)
        rerank_acc, rerank_lat = run_reranking(client, rerank_ds)
        cost = 0.001
        score_link = composite_score(acc, json_val, lat, cost)
        score_rerank = composite_score(rerank_acc, 1.0, rerank_lat, cost * 0.5)
        results[f"{cand.key}:link"] = TrackResult(cand.key, acc, json_val, lat, cost, score_link)
        results[f"{cand.key}:rerank"] = TrackResult(
            cand.key, rerank_acc, 1.0, rerank_lat, cost * 0.5, score_rerank
        )
        click.echo(f"{cand.key}: linking={acc:.2f} rerank={rerank_acc:.2f} score={score_link:.3f}")

    winners = {
        "example_linking": "deepseek-chat",
        "rerank": "gemini-flash",
        "synthesis": "deepseek-chat",
    }
    if results:
        link_best = max(
            (r for k, r in results.items() if k.endswith(":link")),
            key=lambda r: r.score,
            default=None,
        )
        rerank_best = max(
            (r for k, r in results.items() if k.endswith(":rerank")),
            key=lambda r: r.score,
            default=None,
        )
        if link_best:
            winners["example_linking"] = link_best.model_key
            winners["synthesis"] = link_best.model_key
        if rerank_best:
            winners["rerank"] = rerank_best.model_key

    output_data = {
        "tasks": winners,
        "models": {
            c.key: {"model_id": c.model_id, "base_url": c.base_url} for c in DEFAULT_CANDIDATES
        },
        "last_run": {
            k: {
                "accuracy": v.accuracy,
                "json_validity": v.json_validity,
                "latency_ms": v.avg_latency_ms,
                "score": v.score,
            }
            for k, v in results.items()
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        yaml.dump(output_data, f, allow_unicode=True, default_flow_style=False)
    click.echo(f"Results written to {out_path}")


if __name__ == "__main__":
    main()
