"""Verify documentation-only hybrid retrieval, routing, and fallbacks."""

import json
from pathlib import Path
from unittest.mock import ANY, patch

import pytest
import torch
from torch import Tensor

from src.__main__ import main
from src.ingestion import Chunk
from src.retrieval.bm25 import BM25Document, BM25Index
from src.retrieval.mode import resolve_retrieval_mode
from src.retrieval.semantic import (
    build_semantic_index,
    is_documentation_document,
)
from src.retrieval.semantic.workflow import SemanticIndexBuildReport


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("dataset_docs_public.json", "hybrid"),
        ("team-docs.json", "hybrid"),
        ("dataset_code_public.json", "bm25"),
        ("docs_and_code.json", "bm25"),
        ("questions.json", "bm25"),
        ("documentation.json", "bm25"),
    ],
)
def test_auto_mode_follows_the_dataset_file_name(
    name: str,
    expected: str,
) -> None:
    assert resolve_retrieval_mode("auto", Path("any") / name) == expected


def test_auto_mode_ignores_directory_names() -> None:
    path = Path("docs") / "questions.json"
    assert resolve_retrieval_mode("auto", path) == "bm25"


@pytest.mark.parametrize("mode", ["bm25", "hybrid"])
def test_explicit_mode_overrides_the_file_name(mode: str) -> None:
    path = Path("dataset_code_public.json")
    assert resolve_retrieval_mode(mode, path) == mode


def test_unknown_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="Retrieval mode must be one of"):
        resolve_retrieval_mode("semantic", Path("dataset_docs.json"))


def _document(path: str, text: str) -> BM25Document:
    return BM25Document(
        chunk=Chunk(path, 0, len(text), text),
        content_terms=(text,),
    )


class _RecordingEncoder:
    def __init__(self) -> None:
        self.texts: tuple[str, ...] = ()

    def encode(self, texts: tuple[str, ...]) -> Tensor:
        self.texts = texts
        return torch.nn.functional.normalize(
            torch.ones(len(texts), 3), dim=1
        )


def test_semantic_index_can_be_limited_to_documentation_chunks() -> None:
    index = BM25Index(
        [
            _document("docs/guide.md", "guide"),
            _document("src/cache.py", "cache"),
            _document("CMakeLists.txt", "cmake"),
        ]
    )
    encoder = _RecordingEncoder()

    semantic = build_semantic_index(
        index, encoder, include=is_documentation_document
    )

    assert encoder.texts == ("guide", "cmake")
    assert [d.chunk.file_path for d in semantic.documents] == [
        "docs/guide.md",
        "CMakeLists.txt",
    ]


def _project(tmp_path: Path) -> None:
    corpus = tmp_path / "data" / "raw"
    corpus.mkdir(parents=True)
    (corpus / "guide.md").write_text(
        "# Cache\n\nThe cache stores chunks.\n", encoding="utf-8"
    )
    (corpus / "cache.py").write_text(
        "def cache():\n    return 'chunks'\n", encoding="utf-8"
    )
    datasets = tmp_path / "data" / "datasets"
    datasets.mkdir(parents=True)
    payload = {"rag_questions": [{"question_id": "q1", "question": "cache"}]}
    for name in ("dataset_docs_x.json", "dataset_code_x.json"):
        (datasets / name).write_text(json.dumps(payload), encoding="utf-8")


def _index(tmp_path: Path) -> None:
    main(["index", "--project_root", str(tmp_path)])


def _search(tmp_path: Path, name: str, *extra: str) -> Path:
    main(
        [
            "search_dataset",
            "--project_root",
            str(tmp_path),
            "--dataset_path",
            f"data/datasets/{name}",
            "--save_directory",
            "out",
            "--k",
            "2",
            *extra,
        ]
    )
    return tmp_path / "out" / name


def test_index_succeeds_with_bm25_when_the_model_is_unavailable(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _project(tmp_path)

    _index(tmp_path)

    captured = capsys.readouterr()
    assert "semantic_index:" in captured.out and "unavailable" in captured.out
    assert "BM25 only" in captured.err
    assert (tmp_path / "data" / "processed" / "bm25-index.json").is_file()
    assert not (tmp_path / "data" / "processed" / "semantic-index").exists()


def test_index_builds_semantic_embeddings_for_documentation_only(
    tmp_path: Path,
) -> None:
    _project(tmp_path)
    report = SemanticIndexBuildReport(1, 384, 0.1, 0.2, 0.0)
    with patch(
        "src.cli.build_and_store_semantic_index", return_value=report
    ) as build:
        _index(tmp_path)

    build.assert_called_once_with(
        ANY,
        tmp_path / "data" / "processed" / "semantic-index",
        ANY,
        corpus_fingerprint=ANY,
        pipeline_fingerprint=ANY,
        include=is_documentation_document,
    )


def test_docs_dataset_falls_back_to_identical_bm25_results(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _project(tmp_path)
    _index(tmp_path)

    auto = _search(tmp_path, "dataset_docs_x.json").read_bytes()
    warning = capsys.readouterr().err
    bm25 = _search(
        tmp_path, "dataset_docs_x.json", "--retrieval_mode", "bm25"
    ).read_bytes()

    assert "using BM25 only" in warning
    assert auto == bm25


def test_code_dataset_never_uses_hybrid_retrieval(tmp_path: Path) -> None:
    _project(tmp_path)
    _index(tmp_path)

    with patch("src.cli.run_stored_hybrid_retrieval") as hybrid:
        _search(tmp_path, "dataset_code_x.json")

    hybrid.assert_not_called()


def test_docs_dataset_uses_default_hybrid_configuration(
    tmp_path: Path,
) -> None:
    _project(tmp_path)
    _index(tmp_path)

    with (
        patch("src.cli.run_stored_hybrid_retrieval") as hybrid,
        patch("src.cli.run_stored_retrieval") as bm25,
    ):
        _search(tmp_path, "dataset_docs_x.json")

    bm25.assert_not_called()
    _, kwargs = hybrid.call_args
    assert hybrid.call_args.args[4:6] == (2, 20)
    assert kwargs["rrf_parameters"].lexical_weight == 10.0
    assert kwargs["rrf_parameters"].semantic_weight == 1.0
    assert kwargs["rrf_parameters"].rank_constant == 60


def test_invalid_retrieval_mode_is_a_clean_cli_error(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _project(tmp_path)
    _index(tmp_path)

    with pytest.raises(SystemExit):
        _search(tmp_path, "dataset_docs_x.json", "--retrieval_mode", "x")

    assert "Retrieval mode must be one of" in capsys.readouterr().err
