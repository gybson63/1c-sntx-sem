"""Tests for configuration loading."""

from __future__ import annotations

from pathlib import Path

import yaml

from sntx_sem.config import AppConfig, EmbeddingConfig, LLMConfig, load_config, resolve_api_key


def test_coerce_embedding_model_openai_name_for_local_provider() -> None:
    from sntx_sem.config import coerce_embedding_model

    model, note = coerce_embedding_model("sentence_transformers", "text-embedding-3-small")
    assert model == "intfloat/multilingual-e5-base"
    assert note is not None


def test_coerce_embedding_model_local_name_for_api_provider() -> None:
    from sntx_sem.config import coerce_embedding_model

    model, note = coerce_embedding_model(
        "openai_compatible",
        "intfloat/multilingual-e5-small",
        previous_provider="sentence_transformers",
    )
    assert model == "text-embedding-3-small"
    assert note is not None


def test_align_embedding_with_index_uses_built_model(tmp_path: Path) -> None:
    from sntx_sem.config import AppConfig, align_embedding_with_index
    from sntx_sem.index.meta import save_index_meta

    index_dir = tmp_path / "index"
    cfg = AppConfig(
        index_dir=index_dir,
        embedding=EmbeddingConfig(
            provider="sentence_transformers",
            model="text-embedding-3-small",
        ),
    )
    save_index_meta(
        index_dir,
        indexed_count=100,
        platform_version="8.3.27",
        embedding_provider="sentence_transformers",
        embedding_model="intfloat/multilingual-e5-small",
        embedding_dimensions=384,
    )
    aligned = align_embedding_with_index(cfg)
    assert aligned.model == "intfloat/multilingual-e5-small"
    assert aligned.query_prefix == "query: "
    assert aligned.passage_prefix == "passage: "


def test_align_embedding_with_index_keeps_config_when_in_sync(tmp_path: Path) -> None:
    from sntx_sem.config import AppConfig, align_embedding_with_index
    from sntx_sem.index.meta import save_index_meta

    index_dir = tmp_path / "index"
    cfg = AppConfig(
        index_dir=index_dir,
        embedding=EmbeddingConfig(
            provider="sentence_transformers",
            model="intfloat/multilingual-e5-small",
        ),
    )
    save_index_meta(
        index_dir,
        indexed_count=100,
        platform_version="8.3.27",
        embedding_provider="sentence_transformers",
        embedding_model="intfloat/multilingual-e5-small",
    )
    aligned = align_embedding_with_index(cfg)
    assert aligned.model == "intfloat/multilingual-e5-small"


def test_resolve_paths_keeps_absolute_docker_paths() -> None:
    cfg = AppConfig(
        hbk_dir=Path("/hbk"),
        data_dir=Path("/data"),
        export_dir=Path("/data/export"),
        index_dir=Path("/data/index"),
    )
    cfg.resolve_paths(Path("/config"))
    assert cfg.hbk_dir == Path("/hbk").resolve()
    assert cfg.data_dir == Path("/data").resolve()
    assert cfg.export_dir == Path("/data/export").resolve()


def test_resolve_paths_resolves_relative_to_config_dir() -> None:
    cfg = AppConfig()
    cfg.resolve_paths(Path("/config"))
    assert cfg.hbk_dir == (Path("/config") / "hbk").resolve()
    assert cfg.data_dir == (Path("/config") / "data").resolve()


def test_resolve_api_key_from_config_value() -> None:
    assert resolve_api_key("secret-key", "OPENAI_API_KEY") == "secret-key"


def test_resolve_api_key_from_env(monkeypatch) -> None:
    monkeypatch.setenv("MY_KEY", "from-env")
    assert resolve_api_key("", "MY_KEY") == "from-env"


def test_embedding_resolved_api_key_prefers_config() -> None:
    cfg = EmbeddingConfig(api_key="inline", api_key_env="OPENAI_API_KEY")
    assert cfg.resolved_api_key == "inline"


def test_llm_resolved_api_key_from_env(monkeypatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "ds-key")
    cfg = LLMConfig(api_key_env="DEEPSEEK_API_KEY")
    assert cfg.resolved_api_key == "ds-key"


def test_index_embedding_mismatch_when_model_differs() -> None:
    from sntx_sem.config import AppConfig, index_embedding_mismatch

    cfg = AppConfig(
        embedding=EmbeddingConfig(
            provider="openai_compatible",
            base_url="https://polza.ai/api/v1",
            model="text-embedding-3-small",
        )
    )
    index_meta = {"embedding_model": "intfloat/multilingual-e5-small"}
    assert index_embedding_mismatch(cfg, index_meta, indexed_count=100) is True


def test_index_embedding_mismatch_false_when_in_sync() -> None:
    from sntx_sem.config import AppConfig, index_embedding_mismatch

    cfg = AppConfig(
        embedding=EmbeddingConfig(
            provider="openai_compatible",
            model="text-embedding-3-small",
        )
    )
    index_meta = {
        "embedding_provider": "openai_compatible",
        "embedding_model": "text-embedding-3-small",
    }
    assert index_embedding_mismatch(cfg, index_meta, indexed_count=100) is False


def test_load_config_search_section(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        yaml.safe_dump(
            {"search": {"dense_top_k": 15, "rrf_k": 40, "build_vector_index": False}},
            allow_unicode=True,
        ),
        encoding="utf-8",
    )

    cfg = load_config(config_file)
    assert cfg.search.dense_top_k == 15
    assert cfg.search.rrf_k == 40
    assert cfg.search.final_top_k == 5
    assert cfg.search.build_vector_index is False


def test_collect_database_issues_export_missing(tmp_path: Path) -> None:
    from sntx_sem.config import collect_database_issues

    issues = collect_database_issues(
        ready=False,
        chunks_file=tmp_path / "missing.jsonl",
        meta_file=tmp_path / "chunks_meta.json",
        lance_dir=tmp_path / "help_chunks.lance",
        export_count=0,
        indexed_count=0,
        embedding_mismatch=False,
        partial_index=False,
        config_model="intfloat/multilingual-e5-base",
        index_model=None,
    )
    codes = {item["code"] for item in issues}
    assert "export_missing" in codes
    assert "not_ready" not in codes


def test_collect_database_issues_index_broken(tmp_path: Path) -> None:
    from sntx_sem.config import collect_database_issues

    meta_file = tmp_path / "chunks_meta.json"
    meta_file.write_text("{}", encoding="utf-8")
    issues = collect_database_issues(
        ready=False,
        chunks_file=tmp_path / "all_chunks.jsonl",
        meta_file=meta_file,
        lance_dir=tmp_path / "help_chunks.lance",
        export_count=100,
        indexed_count=50,
        embedding_mismatch=False,
        partial_index=True,
        config_model="intfloat/multilingual-e5-base",
        index_model="intfloat/multilingual-e5-small",
    )
    codes = {item["code"] for item in issues}
    assert "index_broken" in codes
    assert "partial_index" in codes


def test_collect_database_issues_bsp_not_indexed(tmp_path: Path) -> None:
    from sntx_sem.config import collect_database_issues

    meta_file = tmp_path / "chunks_meta.json"
    meta_file.write_text(
        '[{"id": "platform:1", "domain": "platform_api", "search_text": "test"}]',
        encoding="utf-8",
    )
    lance_dir = tmp_path / "help_chunks.lance"
    lance_dir.mkdir()
    chunks_file = tmp_path / "all_chunks.jsonl"
    chunks_file.write_text('{"id": "platform:1", "domain": "platform_api"}\n', encoding="utf-8")

    issues = collect_database_issues(
        ready=True,
        chunks_file=chunks_file,
        meta_file=meta_file,
        lance_dir=lance_dir,
        export_count=1,
        indexed_count=1,
        embedding_mismatch=False,
        partial_index=False,
        config_model="intfloat/multilingual-e5-base",
        index_model="intfloat/multilingual-e5-base",
        bsp_expected=True,
        bsp_indexed_count=0,
    )
    codes = {item["code"] for item in issues}
    assert "bsp_not_indexed" in codes


def test_estimate_indexed_chunks_from_build_meta(tmp_path: Path) -> None:
    from sntx_sem.config import estimate_indexed_chunks
    from sntx_sem.index.meta import load_index_meta, save_index_meta

    index_dir = tmp_path / "index"
    index_dir.mkdir()
    save_index_meta(
        index_dir,
        indexed_count=42,
        platform_version="8.3.27",
        embedding_provider="sentence_transformers",
        embedding_model="intfloat/multilingual-e5-small",
    )
    index_meta = load_index_meta(index_dir)
    assert estimate_indexed_chunks(index_dir, index_meta) == 42


def test_estimate_indexed_chunks_streams_meta_file(tmp_path: Path) -> None:
    from sntx_sem.config import estimate_indexed_chunks

    index_dir = tmp_path / "index"
    index_dir.mkdir()
    meta_file = index_dir / "chunks_meta.json"
    meta_file.write_text('{"id": "a"}\n{"id": "b"}\n', encoding="utf-8")
    assert estimate_indexed_chunks(index_dir, {}) == 2


def test_bundled_status_detects_corrupt_lance(tmp_path: Path) -> None:
    from sntx_sem.config import AppConfig, bundled_database_status
    from sntx_sem.index.meta import save_index_meta

    export_dir = tmp_path / "export"
    index_dir = tmp_path / "index"
    export_dir.mkdir()
    index_dir.mkdir()
    (export_dir / "all_chunks.jsonl").write_text('{"id":"x"}\n', encoding="utf-8")
    (index_dir / "chunks_meta.json").write_text("[{}]", encoding="utf-8")
    (index_dir / "help_chunks.lance").mkdir()
    save_index_meta(
        index_dir,
        indexed_count=1,
        platform_version="8.3.27",
        embedding_provider="sentence_transformers",
        embedding_model="intfloat/multilingual-e5-small",
        vector_index_built=False,
    )

    cfg = AppConfig(export_dir=export_dir, index_dir=index_dir)
    status = bundled_database_status(cfg)
    assert status["ready"] is False
    assert status["index"]["lance_ok"] is False
    codes = {item["code"] for item in status["issues"]}
    assert "index_broken" in codes
