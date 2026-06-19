"""CLI entry point."""

from __future__ import annotations

import json
from pathlib import Path

import click

from sntx_sem.config import detect_platform_path, load_config, update_manifest_indexed_count
from sntx_sem.examples.linker import link_examples_batch
from sntx_sem.examples.scanner import (
    export_examples_jsonl,
    keyword_link_examples,
    scan_config_path,
)
from sntx_sem.examples.store import ExamplesStore
from sntx_sem.hbk.extractor import ingest_hbk_dir
from sntx_sem.hbk.java_bridge import merge_java_export, run_java_exporter
from sntx_sem.index.store import EmbeddingModel, HelpIndex


@click.group()
def main() -> None:
    """1C syntax help semantic search toolkit."""


def _run_ingest(
    hbk_path: Path,
    version: str,
    platform_path: str | None,
) -> None:
    cfg = load_config()

    if platform_path:
        src = Path(platform_path)
        hbk_path.mkdir(parents=True, exist_ok=True)
        for name in [
            "shcntx_ru.hbk",
            "shcntx_root.hbk",
            "shlang_ru.hbk",
            "shlang_root.hbk",
            "shquery_ru.hbk",
            "shquery_root.hbk",
        ]:
            src_file = src / "bin" / name if (src / "bin").is_dir() else src / name
            if src_file.is_file():
                (hbk_path / name).write_bytes(src_file.read_bytes())
                click.echo(f"Copied {name}")

    if not hbk_path.is_dir() or not list(hbk_path.glob("*.hbk")):
        auto = detect_platform_path()
        if auto:
            click.echo(f"Auto-detected platform: {auto}")
            _run_ingest(hbk_path, version, str(auto))
            return
        raise click.ClickException(f"No HBK files in {hbk_path}")

    cfg.export_dir.mkdir(parents=True, exist_ok=True)
    stats = ingest_hbk_dir(hbk_path, version, cfg.export_dir)
    click.echo(f"Python ingest: {stats}")

    if cfg.java_exporter.enabled:
        jar = Path(cfg.java_exporter.jar_path)
        java_chunks = run_java_exporter(jar, hbk_path, cfg.export_dir, version)
        if java_chunks:
            merge_java_export(java_chunks, cfg.export_dir)
            click.echo(f"Java exporter: {len(java_chunks)} chunks")
        else:
            click.echo("Java exporter skipped (JAR missing or shcntx not found)")


@main.command("ingest")
@click.option("--hbk-dir", type=click.Path(exists=True), default=None)
@click.option("--platform-version", default=None)
@click.option("--platform-path", type=click.Path(exists=True), default=None)
def ingest_cmd(
    hbk_dir: str | None, platform_version: str | None, platform_path: str | None
) -> None:
    """Extract help chunks from HBK files."""
    cfg = load_config()
    hbk_path = Path(hbk_dir) if hbk_dir else cfg.hbk_dir
    version = platform_version or cfg.platform_version
    _run_ingest(hbk_path, version, platform_path)


@main.command("index")
@click.option("--rebuild/--no-rebuild", default=True)
@click.option("--chunks", type=click.Path(exists=True), default=None)
@click.option("--domain", default=None, help="Index only chunks from domain (e.g. query_lang)")
def index_cmd(rebuild: bool, chunks: str | None, domain: str | None) -> None:
    """Build vector index from exported chunks."""
    cfg = load_config()
    jsonl = Path(chunks) if chunks else cfg.export_dir / "all_chunks.jsonl"
    if not jsonl.is_file():
        raise click.ClickException(f"Chunks not found: {jsonl}. Run ingest first.")

    embedder = EmbeddingModel(cfg.embedding.model, cfg.embedding.device)
    index = HelpIndex(cfg.index_dir, embedder, cfg.search)
    raw_chunks = index.load_chunks_from_jsonl(jsonl)
    if domain:
        raw_chunks = [c for c in raw_chunks if c.get("domain") == domain]
        click.echo(f"Filtered to domain={domain}: {len(raw_chunks)} chunks")
    count = index.build(raw_chunks, rebuild=rebuild)
    update_manifest_indexed_count(cfg.data_dir, count)
    click.echo(f"Indexed {count} chunks -> {cfg.index_dir}")


@main.command("status")
def status_cmd() -> None:
    """Show local database readiness."""
    from sntx_sem.config import bundled_database_status

    cfg = load_config()
    status = bundled_database_status(cfg)
    click.echo(json.dumps(status, ensure_ascii=False, indent=2))
    if status["ready"]:
        click.echo("Database is ready — connect MCP and search.")
    elif status.get("partial_index"):
        click.echo(
            f"Partial index ({status['indexed_chunks']}/{status['export_chunks']}). "
            "Run: python -m sntx_sem index --rebuild"
        )
    else:
        click.echo(
            "Database not ready. Build locally from your 1C platform install:\n"
            '  python -m sntx_sem ingest --platform-path "C:/Program Files/1cv8/.../bin"\n'
            "  python -m sntx_sem index --rebuild"
        )


@main.command("search")
@click.argument("query")
@click.option("--domain", default="all")
@click.option("--limit", default=5, type=int)
def search_cmd(query: str, domain: str, limit: int) -> None:
    """Search help index from CLI."""
    cfg = load_config()
    embedder = EmbeddingModel(cfg.embedding.model, cfg.embedding.device)
    index = HelpIndex(cfg.index_dir, embedder, cfg.search)
    results = index.search(query, domain=domain, limit=limit)
    for i, r in enumerate(results, 1):
        click.echo(f"{i}. [{r.domain}] {r.title} (score={r.score:.4f})")
        click.echo(f"   id={r.id}")


@main.command("scan-examples")
@click.option("--link/--no-link", default=False, help="Use LLM linking (requires API key)")
def scan_examples_cmd(link: bool) -> None:
    """Scan local configs for code examples."""
    cfg = load_config()
    cache_dir = cfg.data_dir / ".sntx_sem_cache"
    all_examples = []
    for lc in cfg.local_configs:
        path = Path(lc.path)
        if not path.is_dir():
            click.echo(f"Skip missing config: {path}")
            continue
        found = scan_config_path(path, lc.label, cache_dir)
        click.echo(f"{lc.label}: {len(found)} examples")
        all_examples.extend(found)

    jsonl = cfg.export_dir / "all_chunks.jsonl"
    topics = []
    if jsonl.is_file():
        with jsonl.open(encoding="utf-8") as f:
            topics = [json.loads(line) for line in f if line.strip()]

    if link and all_examples and topics:
        all_examples = link_examples_batch(all_examples, topics)
        click.echo(f"LLM linked: {len(all_examples)} examples")
    elif all_examples and topics:
        all_examples = keyword_link_examples(all_examples, topics)
        click.echo(f"Keyword linked: {len(all_examples)} examples")

    out = cfg.data_dir / "examples.jsonl"
    export_examples_jsonl(all_examples, out)
    ExamplesStore(out).save(all_examples)
    click.echo(f"Saved {len(all_examples)} examples -> {out}")


if __name__ == "__main__":
    main()
